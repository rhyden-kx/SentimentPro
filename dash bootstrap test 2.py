import os
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
from datetime import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Sample data for the Issues page
topics = ['Topic {}'.format(i) for i in range(1, 11)]
issues = {
    topic: ['Issue {}'.format(i) for i in range(1, 6)]
    for topic in topics
}
solutions = {
    topic: ['Solution {}'.format(i) for i in range(1, 6)]
    for topic in topics
}

# Define function to process CSV files
def process_csv(csv_file):
    # Read the CSV file
    topic_df = pd.read_csv(csv_file)
    
    # Initialize the SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Initialize lists to store data
    review_texts = []
    positive_scores = []
    negative_scores = []
    neutral_scores = []
    compound_scores = []
    nps_indiv = []
    nps_category = []  # New column for NPS categories
    
    # Perform sentiment analysis and store scores in lists
    for review in topic_df['review']:
        vs = analyzer.polarity_scores(review)
        
        review_texts.append(review)
        positive_scores.append(vs['pos'])
        negative_scores.append(vs['neg'])
        neutral_scores.append(vs['neu'])
        compound_scores.append(vs['compound'])
        
        # Map compound scores to nps_indiv based on specified intervals
        if -1 <= vs['compound'] <= -9/11:
            nps_indiv.append(0)
        elif -9/11 < vs['compound'] <= -7/11:
            nps_indiv.append(1)
        elif -7/11 < vs['compound'] <= -5/11:
            nps_indiv.append(2)
        elif -5/11 < vs['compound'] <= -3/11:
            nps_indiv.append(3)
        elif -3/11 < vs['compound'] <= -1/11:
            nps_indiv.append(4)
        elif -1/11 < vs['compound'] <= 1/11:
            nps_indiv.append(5)
        elif 1/11 < vs['compound'] <= 3/11:
            nps_indiv.append(6)
        elif 3/11 < vs['compound'] <= 5/11:
            nps_indiv.append(7)
        elif 5/11 < vs['compound'] <= 7/11:
            nps_indiv.append(8)
        elif 7/11 < vs['compound'] <= 9/11:
            nps_indiv.append(9)
        else:
            nps_indiv.append(10)
        
        # Map nps_indiv scores to NPS categories
        if nps_indiv[-1] >= 9:  # Promoters
            nps_category.append('Promoter')
        elif nps_indiv[-1] >= 7:  # Passives
            nps_category.append('Passive')
        else:  # Detractors
            nps_category.append('Detractor')
    
    # Add sentiment scores and NPS categories to the DataFrame
    topic_df['positive_scores'] = positive_scores
    topic_df['negative_scores'] = negative_scores
    topic_df['neutral_scores'] = neutral_scores
    topic_df['compound_scores'] = compound_scores
    topic_df['nps_indiv'] = nps_indiv
    topic_df['nps_category'] = nps_category

    # Merge dataframes
    topic_df = date_df.merge(topic_df, on='review', how='right')
    
    # Clean date
    topic_df['date_clean'] = pd.to_datetime(topic_df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    topic_df['date_clean'] = topic_df['date_clean'].combine_first(pd.to_datetime(topic_df['date'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce'))
    topic_df['date_clean'] = topic_df['date_clean'].astype(str)
    topic_df['date_clean'] = topic_df['date_clean'].str[:10]
    topic_df['date_clean'] = pd.to_datetime(topic_df['date_clean'])

    topic_df = topic_df.drop(['date', 'Unnamed: 0'], axis=1)
    topic_df = topic_df.rename(columns={'date_clean':'Date'})    
    return topic_df

# Define function to calculate NPS for a topic
def topic_nps(topic_df, start_date, end_date):
    # Filter by date
    filtered_df = topic_df[(topic_df['Date'] >= start_date) & (topic_df['Date'] <= end_date)]
    
    # Count the occurrences of each label
    label_counts = filtered_df['nps_category'].value_counts()

    # Calculate Net Promoter Score (NPS)
    promoter_count = label_counts.get('Promoter', 0)
    detractor_count = label_counts.get('Detractor', 0)
    passive_count = label_counts.get('Passive', 0)
    total_count = promoter_count + detractor_count + passive_count

    # Calculate NPS
    if total_count == 0:
        nps = None
    else:
        nps = ((promoter_count - detractor_count) / total_count) * 100
        nps = round(nps, 2)
    
    return nps

# Define function to calculate NPS for each issue within a topic
def issue_nps(topic_df, start_date, end_date):
    # Filter by date
    filtered_df = topic_df[(topic_df['Date'] >= start_date) & (topic_df['Date'] <= end_date)]
    
    unique_keys = topic_df['key'].unique()
    issues_nps_scores = {}

    for key in unique_keys:
        key_df = topic_df[topic_df['key'] == key]
        label_counts = key_df['nps_category'].value_counts()

        promoter_count = label_counts.get('Promoter', 0)
        detractor_count = label_counts.get('Detractor', 0)
        passive_count = label_counts.get('Passive', 0)
        total_count = promoter_count + detractor_count + passive_count

        if total_count == 0:
            issues_nps_scores[key] = None
        else:
            nps = ((promoter_count - detractor_count) / total_count) * 100
            issues_nps_scores[key] = round(nps, 2)
        issuesNPS = pd.DataFrame(list(issues_nps_scores.items()), columns=['Issue', 'NPS'])

    return issuesNPS

# Function to score sentiment using VADER
def nps_score(review) :
    vs = analyzer.polarity_scores(review)
    compound_score = vs['compound']
    
    # Mapping compound scores to NPS categories
    if -1 <= compound_score <= -9/11:
        return 0  # Detractor
    elif -9/11 < compound_score <= 1/11:
        return 1  # Passive
    else:
        return 2  # Promoter

# Load initial data
date_df = pd.DataFrame({'review': ['Initial Review'], 'Date': [pd.to_datetime('2023-01-01')]})
date_df['key'] = 0

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(id="page-content")
])

# Sidebar layout
sidebar = html.Div(
    [
        html.H2("Navigation", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Promoters", href="/promoters", id="promoters-link"),
                dbc.NavLink("Detractors", href="/detectors", id="detectors-link"),
                dbc.NavLink("Passive", href="/passive", id="passive-link"),
                dbc.NavLink("Issues", href="/issues", id="issues-link"),
                dbc.NavLink("NPS by Rater", href="/nps_rater", id="nps-rater-link"),
                dbc.NavLink("Information", href="/information", id="information-link"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar",
)

# Dashboard layout
dashboard_layout = html.Div([
    html.H1("Overall NPS Scores", style={'textAlign': 'center'}),
    dcc.Graph(id='overall-nps-graph'),
    html.Hr(),
    html.H3("Top Issues by NPS", style={'textAlign': 'center'}),
    dcc.Graph(id='top-issues-graph'),
])

# Callback to render different pages based on URL pathname
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/promoters":
        return html.H1("Promoters")
    elif pathname == "/detectors":
        return html.H1("Detectors")
    elif pathname == "/passive":
        return html.H1("Passive")
    elif pathname == "/issues":
        return issues_layout
    elif pathname == "/nps_rater":
        return html.H1("NPS by Rater")
    elif pathname == "/information":
        return html.H1("Information")
    else:
        return dashboard_layout

# Callback to update overall NPS graph
@app.callback(
    Output('overall-nps-graph', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')])
def update_overall_nps(start_date, end_date):
    # Calculate NPS for the overall dataset
    nps = topic_nps(date_df, start_date, end_date)

    # Create a gauge chart for NPS
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=nps,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "NPS"},
        gauge={
            'axis': {'range': [-100, 100]},
            'steps': [
                {'range': [-100, 0], 'color': "red"},
                {'range': [0, 100], 'color': "green"}
            ],
        }
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))

    return fig

# Callback to update top issues graph
@app.callback(
    Output('top-issues-graph', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')])
def update_top_issues(start_date, end_date):
    # Calculate NPS for each issue
    issues_nps = issue_nps(date_df, start_date, end_date)

    # Sort issues by NPS in descending order
    sorted_issues = issues_nps.sort_values(by='NPS', ascending=False)

    # Create a bar chart for top issues by NPS
    fig = px.bar(sorted_issues.head(10), x='Issue', y='NPS', title='Top Issues by NPS')
    fig.update_layout(xaxis_title='Issue', yaxis_title='NPS', margin=dict(l=20, r=20, t=30, b=20))

    return fig

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
