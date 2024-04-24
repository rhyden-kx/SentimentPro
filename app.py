import os
import json
import random
import time
import dash
import base64
import datetime
import io
import dash_bootstrap_components as dbc
from dash import dash_table
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
from dash.dash_table.Format import Group
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash_bootstrap_templates import load_figure_template
from datetime import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_figure_template("darkly")


# Set Directory (For Local version only)

# Read Data

## Dashboard Page:
topics_df = pd.read_csv('/app/data/TopicsofReviews.csv')
nps_df = pd.read_csv('/app/data/nps_df.csv')
score_df = pd.read_csv('/app/data/score_df.csv')
date_df = pd.read_csv('/app/data/combined_data.csv')

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Wrap function to read and edit CSV files
def process_csv(csv_file):
    # Read the CSV file
    # Specify the path to the CSV file
    csv_path = os.path.join('/app/data/', csv_file)
    
    # Read the CSV file
    topic_df = pd.read_csv(csv_path)
    
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

    # merge dfs into data
    topic_df = date_df.merge(topic_df, on='review', how='right')
    
    # clean date
    topic_df['date_clean'] = pd.to_datetime(topic_df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    topic_df['date_clean'] = topic_df['date_clean'].combine_first(pd.to_datetime(topic_df['date'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce'))
    # sum(data['date_clean'].isnull()) # should output 0
    
    topic_df['date_clean'] = topic_df['date_clean'].astype(str)
    topic_df['date_clean'] = topic_df['date_clean'].str[:10]
    topic_df['date_clean'] = pd.to_datetime(topic_df['date_clean'])

    topic_df = topic_df.drop(['date', 'Unnamed: 0'], axis=1)
    topic_df = topic_df.rename(columns={'date_clean':'Date'})    
    return topic_df


def topic_nps(topic_df, start_date, end_date):
    # filter by date
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


#testing - check df and functions
#process_csv('App Responsiveness.csv').head()
#data = process_csv('App Responsiveness.csv')
#data.shape
#topic_nps(data, data['Date'].min(), data['Date'].max())

def issue_nps(topic_df, start_date, end_date):
    # filter by date
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


# test out the functions
#issue_df = issue_nps(data, data['Date'].min(), data['Date'].max())
#issue_df

# list out csv file names
topics_csv = ['App Responsiveness.csv', 'Competition.csv', 'Credit card usage.csv', 'Customer Services.csv', 'Customer trust.csv', 
              'Login & Account Setup.csv', 'Money Growth (Interest Rates).csv', 'Safety.csv', 'Services & Products.csv', 'User Interface.csv']

# list out topic names
topics = ['App Responsiveness', 'Competition', 'Credit card usage', 'Customer Services', 'Customer trust', 'Login & Account Setup', 
          'Money Growth (Interest Rates)', 'Safety', 'Services & Products', 'User Interface']


#Function to return data for 3 NPS categories
def plot_category_frequency_by_topic(cat):
    # Filter data based on the specified NPS category
    filtered_df = score_df[score_df['nps_category'] == cat]
    filtered_df = filtered_df[filtered_df['Topic_Number'].astype(str).str.len() == 1]

    # Group by Topic_Name and count the frequency of each topic
    topic_freq = filtered_df.groupby('Topic_Name').size().reset_index(name='Frequency')

    # Sort the DataFrame by the 'Frequency' column in descending order
    topic_freq_sorted = topic_freq.sort_values(by='Frequency', ascending=False)

    # Determine the range of frequencies
    min_freq = topic_freq_sorted['Frequency'].min()
    max_freq = topic_freq_sorted['Frequency'].max()

    # Define the custom color scale
    colorscale = [[0, '#e0c7ff'], [1, '#5200b8']]
    if cat == "Detractor":
        colorscale = [[0, '#5200b8'], [1, '#e0c7ff']]

    # Plot the bar chart
    fig = px.bar(topic_freq_sorted, x='Topic_Name', y='Frequency',
                 title=f'Frequency of Reviews for Category: {cat}',
                 labels={'Frequency': 'Number of Reviews'},
                 color='Frequency', color_continuous_scale=colorscale,
                 range_color=[min_freq, max_freq])

    fig.update_layout(xaxis_title='Topic', yaxis_title='Frequency')

    return fig


def table_reviews_category(cat, selected_topic):
    # Filter data based on the specified NPS category
    filtered_df = score_df[score_df['nps_category'] == cat]

    # Filter data based on selected topic
    topic_filtered_df = filtered_df[filtered_df['Topic_Name'] == selected_topic]
    # Extract review text
    reviews = topic_filtered_df['Review'].tolist()

    table = dash_table.DataTable(
        id='table',
        columns=[{'name': 'Review', 'id': 'Review'}],
        data=[{'Review': review} for review in reviews],
        style_table={'borderRadius': '15px'},  # Dark background color
        style_header={'backgroundColor': '#6a05ed', 'fontWeight': 'bold', 'color': '#fafaf9'},  # Purple header text
        style_cell={'backgroundColor': '#303030', 'textAlign': 'left', 'padding': '5px', 'color': '#fafaf9', 'whiteSpace': 'normal'}  
        )
    return table


# Issues Page
def plot_default_graph():
    # Merge the dataframes
    all_data = data.merge(topic_df_issues, on='review', how='inner')

    # Clean the date
    all_data['date_clean'] = pd.to_datetime(all_data['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    all_data['date_clean'] = all_data['date_clean'].combine_first(pd.to_datetime(all_data['date'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce'))
    all_data['date_clean'] = all_data['date_clean'].astype(str).str[:10]
    all_data['date_clean'] = pd.to_datetime(all_data['date_clean']).dt.date

    # Select columns needed
    df = all_data[['review', 'date_clean', 'Topic_Name', 'Topic_Number']]
    get_topic_name = {
        0: 'App Responsiveness', 1: 'Money Growth (Interest Rates)', 2: 'Customer Services',
        3: 'Services & Products', 4: 'User Interface', 5: 'Credit card',
        6: 'Login & Account Setup', 7: 'Competition', 8: 'Safety', 9: 'Customer trust'
    }

    # Split reviews with multiple topics into duplicates of single topics
    df['Topic_Number'] = df['Topic_Number'].astype(str)
    df2 = df[df['Topic_Number'].str.contains(',', regex=False)].copy()

    # Hard coding cos splitting by regex is killing me
    df['Topic_Number'] = df['Topic_Number'].str[0]
    df['Topic_Number'] = df['Topic_Number'].astype(int)
    df['Topic_Name'] = df['Topic_Number'].map(get_topic_name)

    df2['Topic_Number'] = df2['Topic_Number'].str[2]
    df2['Topic_Number'] = df2['Topic_Number'].astype(int)
    df2['Topic_Name'] = df2['Topic_Number'].map(get_topic_name)

    df = pd.concat([df, df2])

    # Group by topic, number, and date
    df = df.groupby(['Topic_Name', 'Topic_Number', 'date_clean']).size().reset_index(name='num_reviews')

    # Select date range
    s = df.date_clean.iloc[5]
    e = df.date_clean.iloc[10]

    # Create the default graph
    topics_over_time = px.line(df, 'date_clean', 'num_reviews', color='Topic_Name',
                               labels={"date_clean": "Date", "num_reviews": "Number of Reviews", "Topic_Name": "Topic"},
                               title="Number of Reviews by Topics over time")
    # Update date range
    topics_over_time.update_xaxes(range=[s, e])

    return topics_over_time

def get_date_range():
    # Merge the dataframes
    all_data = data.merge(topic_df_issues, on='review', how='inner')

    # Clean the date
    all_data['date_clean'] = pd.to_datetime(all_data['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    all_data['date_clean'] = all_data['date_clean'].combine_first(pd.to_datetime(all_data['date'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce'))
    all_data['date_clean'] = all_data['date_clean'].astype(str).str[:10]

    min_date = all_data.date_clean.min()
    max_date = all_data.date_clean.max()
    return [min_date, max_date]

app_responsiveness = pd.read_csv('/app/data/App Responsiveness.csv')
competition = pd.read_csv('/app/data/Competition.csv')
credit_card = pd.read_csv('/app/data/Credit card.csv')
customer_service = pd.read_csv('/app/data/Customer Services.csv')
customer_trust = pd.read_csv('/app/data/Customer trust.csv')
login_account = pd.read_csv('/app/data/Login & Account Setup.csv')
money_growth = pd.read_csv('/app/data/Money Growth (Interest Rates).csv')
safety = pd.read_csv('/app/data/Safety.csv')
service_products = pd.read_csv('/app/data/Services & Products.csv')
user_interface = pd.read_csv('/app/data/User Interface.csv')
data = pd.read_csv('/app/data/combined_data.csv')
solutions_df = pd.read_csv('/app/data/Solutions.csv')

# Test
topics_issues = ['', 'App Responsiveness', 'Competition', 'Credit card usage', 'Customer Services', 'Customer Trust',
          'Login & Account Setup', 'Money Growth (Interest Rates)', 'Safety', 'Service Products', 'User Interface']

datasets = {
    'App Responsiveness': app_responsiveness,
    'Competition': competition,
    'Credit card usage': credit_card,
    'Customer Services': customer_service,
    'Customer Trust': customer_trust,
    'Login & Account Setup': login_account,
    'Money Growth (Interest Rates)': money_growth,
    'Safety': safety,
    'Service Products': service_products,
    'User Interface': user_interface
}

# Issue method
def issue(data, df):
    # Merge the two DataFrames on the 'review' column
    merged = pd.merge(data, df, on='review', how='inner')
    # Drop the 'Unnamed: 0' column
    merged.drop(columns=['Unnamed: 0'], inplace=True)
    # Rename the 'key' column to 'issue'
    merged.rename(columns={'key': 'issue'}, inplace=True)
    # Convert 'date' column to datetime format
    merged['date'] = pd.to_datetime(merged['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # If the previous conversion fails, try a different format
    merged['date'] = merged['date'].combine_first(pd.to_datetime(merged['date'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce'))
    # Convert 'date' column to string
    merged['date'] = merged['date'].astype(str)
    # Extract date only (YYYY-MM-DD)
    merged['date_only'] = merged['date'].str[:10]
    grouped_data = merged.groupby(['date_only', 'issue']).size().reset_index(name='count')
    # Convert 'date_only' to datetime if it's not already in datetime format
    grouped_data['date_only'] = pd.to_datetime(grouped_data['date_only'])
    # Aggregate data by month and issue
    grouped_data['month_year'] = grouped_data['date_only'].dt.to_period('M')
    monthly_data = grouped_data.groupby(['month_year', 'issue']).size().reset_index(name='count')
    # Calculate total count for each issue
    issue_totals = monthly_data.groupby('issue')['count'].sum().sort_values(ascending=False)
    # Select top n issues
    top_issues = issue_totals.head(5).index
    return top_issues 

# Initialize an empty dictionary to store the top issues for each topic
issues = {}

# Call preprocess for each key in the datasets dictionary
for topic, df in datasets.items():
    top_issues = issue(data, df)
    issues[topic] = top_issues


def preprocess(data, df):
    # Merge the two DataFrames on the 'review' column
    merged = pd.merge(data, df, on='review', how='inner')

    # Drop the 'Unnamed: 0' column
    merged.drop(columns=['Unnamed: 0'], inplace=True)

    # Rename the 'key' column to 'issue'
    merged.rename(columns={'key': 'issue'}, inplace=True)

    # Convert 'date' column to datetime format
    merged['date'] = pd.to_datetime(merged['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # If the previous conversion fails, try a different format
    merged['date'] = merged['date'].combine_first(pd.to_datetime(merged['date'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce'))

    # Convert 'date' column to string
    merged['date'] = merged['date'].astype(str)

    # Extract date only (YYYY-MM-DD)
    merged['date_only'] = merged['date'].str[:10]

    return merged

def plot_top_n_issues_time_series(merged_data):
    # Create a time series line plot
    grouped_data = merged_data.groupby(['date_only', 'issue']).size().reset_index(name='count')

    # Convert 'date_only' to datetime if it's not already in datetime format
    grouped_data['date_only'] = pd.to_datetime(grouped_data['date_only'])

    # Aggregate data by month and issue
    grouped_data['month_year'] = grouped_data['date_only'].dt.to_period('M')
    monthly_data = grouped_data.groupby(['month_year', 'issue']).size().reset_index(name='count')

    # Calculate total count for each issue
    issue_totals = monthly_data.groupby('issue')['count'].sum().sort_values(ascending=False)

    # Select top n issues
    top_issues = issue_totals.head(5).index

    # Filter monthly_data for top n issues
    monthly_data_top = monthly_data[monthly_data['issue'].isin(top_issues)]

    # Convert 'month_year' to string format
    monthly_data_top.loc[:, 'month_year'] = monthly_data_top['month_year'].astype(str)

    issue_color_map = {
        'App Responsiveness': 'blue',
        'Competition': 'orange',
        'Credit card usage': 'green',
        'Customer Services': 'red',
        'Customer Trust': 'purple',
        'Login & Account Setup': 'yellow',
        'Money Growth (Interest Rates)': 'cyan',
        'Safety': 'magenta',
        'Service Products': 'teal',
        'User Interface': 'pink'
    }

    # Create a time series line plot
    fig = px.line(monthly_data_top, x='month_year', y='count', color='issue', title=f'Frequency of Top 5 Subtopics Over Time', color_discrete_map=issue_color_map)
    
    # Add markers to the lines
    for trace in fig.data:
        trace.update(mode='lines+markers')
    fig.update_xaxes(title_text='Month')
    fig.update_yaxes(title_text='Number of Reviews')
    fig.update_yaxes(fixedrange=True)
    return fig


def select_related_solutions(topic, df):
    # Filter the DataFrame based on the provided topic
    selected_df = df[df['Topic'] == topic]
    # Drop the 'Topic' column as it's no longer needed
    selected_df = selected_df.drop(columns=['Topic'])
    return selected_df


# NPS Rater Page

API_KEY = "sk-ms7SU43E34tS9UJks5RD2KM3m1JumOR2pM73Dk95VzKjM6TZ"

API_KEY = API_KEY or os.getenv("H2O_GPT_E_API_KEY")

if not API_KEY:
    raise ValueError("Please configure h2ogpte API key")

REMOTE_ADDRESS = "https://h2ogpte.genai.h2o.ai"

from h2ogpte import H2OGPTE

client = H2OGPTE(address=REMOTE_ADDRESS, api_key=API_KEY)

#data extraction
def review_analysis(review):
    extract = client.extract_data(
        text_context_list= [review],
        #pre_prompt_extract="Pay attention and look at all people. Your job is to collect their names.\n",
        prompt_extract="List the good thing and suggestions for improvement. Ignore grammatical errors and awkward languages"
    )
    # List of LLM answers per text input
    extracted_text_list = ''
    for extract_list_item in extract.content:
        for s in extract_list_item.split("\n"):
            extracted_text_list += s + '\n\n'
    return(extracted_text_list)

analyzer = SentimentIntensityAnalyzer()
def nps_score(review):
    vs = analyzer.polarity_scores(review)
    nps_indiv = round((vs['compound'] + 1) * 5)
    return min(max(nps_indiv, 0), 10)

def nps_cat(review):
    nps_indiv = nps_score(review)
    if nps_indiv >= 9:
        return 'Promoter'
    elif nps_indiv >= 7:
        return 'Passive'
    else:
        return 'Detractor'



# DASH APP

dbc_css = (
    "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.4/dbc.min.css"
)
# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc_css, dbc.icons.BOOTSTRAP], suppress_callback_exceptions=True)

# Sidebar
sidebar = html.Div(
    [
        html.Img(src="https://github.com/rhyden-kx/SentimentPro/blob/main/data/sentimentpro-high-resolution-logo-transparent.png?raw=true", style={'width': '100%', 'height': 'auto'}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("NPS", href="/", active="exact"),
                dbc.NavLink("Trends", href="/trends", active="exact", style={"color": "#FFFFFF"}),  # Background color changed to #40268a when active
                dbc.NavLink("Review Rater", href="/review_rater", active="exact", style={"color": "#FFFFFF"}),  # Background color changed to #40268a when active
                dbc.NavLink("Credits", href="/credits", active="exact", style={"color": "#FFFFFF"}),  # Background color changed to #40268a when active
            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar",
    className="bg-dark",  # Color of the entire sidebar changed to #40268a
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem",
        "transition": "0.5s",
        "color": "#FFFFFF"  # Text color of the entire sidebar changed to white
    },
)

# Main content
content = html.Div(
    [
        dcc.Location(id="url"),
        html.Div(id="page-content", style={"margin-left": "18rem", "margin-right": "2rem", "padding-top": "4rem"}),
    ]
)

# NPS Scores page layout
nps_scores_layout = html.Div(
    [
        html.Br(),
        html.H1("Net Promoter Score (NPS) Analysis", style = {"color":"#a142ff"}),
        html.Div(
            dbc.Alert(
                [
                    html.I(className="bi bi-info-circle-fill me-2"),
                    "Click on a bar in the graph below to see the NPS breakdown for that topic.",
                ],
                color="#4a3b78",
                className="d-flex align-items-center",
            ),
        ),
        dbc.Container(
            dcc.DatePickerRange(
                id='date-range-picker',
                start_date='2022-08-01',
                end_date=dt.now().date(),
                display_format='YYYY-MM-DD',
            ),
            fluid=True,
            className="dbc"
        ),
        dcc.Graph(id='nps-by-topic-graph'),
        html.Br(),
        html.Div(id='drill-down-container'),
        dcc.Graph(id='drill-down-graph'),
        html.Br(),
        html.Br(),
    ]
)


# Create layout for Promoter tab
promoter_layout = html.Div(
    [
        html.H4("Promoter Reviews", style={"color": "#a142ff"}),
        html.Div(
            dbc.Alert(
                [
                    html.I(className="bi bi-info-circle-fill me-2"),
                    "Select a topic from the dropdown to view Promoter Reviews for that topic",
                ],
                color="#4a3b78",
                className="d-flex align-items-center",
            ),
        ),
        dbc.Container(
            dcc.Dropdown(
                id="promoter-topic-dropdown",
                options=[{"label": topic, "value": topic} for topic in topics_issues],
                value=topics_issues[0],
                style = {"width":"50%"},
                placeholder = "Select Topic...",
            ),
            fluid=True,
            className="dbc"
        ),
        html.Br(),
        dbc.Container(
            html.Div(id="promoter-review-table"),
            fluid=True,
            className="dbc"
        ),
        html.Br(),
        html.Br(),
        html.Br()
    ]
)

# Create layout for Detractor tab
detractor_layout = html.Div(
    [
        html.H4("Detractor Reviews", style={"color": "#a142ff"}),
        html.Div(
            dbc.Alert(
                [
                    html.I(className="bi bi-info-circle-fill me-2"),
                    "Select a topic from the dropdown to view Detractor Reviews for that topic",
                ],
                color="#4a3b78",
                className="d-flex align-items-center",
            ),
        ),
        dbc.Container(
            dcc.Dropdown(
                id="detractor-topic-dropdown",
                options=[{"label": topic, "value": topic} for topic in topics_issues],
                value=topics_issues[0],
                style = {"width":"50%"},
                placeholder= "Select Topic..."
            ),
            fluid=True,
            className="dbc"
        ),
        html.Br(),
        dbc.Container(
            html.Div(id="detractor-review-table"),
            fluid=True,
            className="dbc"
        ),
        html.Br(),
        html.Br(),
        html.Br()
    ]
)

# Create layout for Passive tab
passive_layout = html.Div(
    [
        html.H4("Passive reviews", style={"color": "#a142ff"}),
        html.Div(
            dbc.Alert(
                [
                    html.I(className="bi bi-info-circle-fill me-2"),
                    "Select a topic from the dropdown to view Passive Reviews for that topic",
                ],
                color="#4a3b78",
                className="d-flex align-items-center",
            ),
        ),
        dbc.Container(
            dcc.Dropdown(
                id="passive-topic-dropdown",
                options=[{"label": topic, "value": topic} for topic in topics_issues],
                value=topics_issues[0],
                style = {"width":"50%"},
                placeholder = "Select Topic..."
            ),
            fluid=True,
            className="dbc"
        ),
        html.Br(),
        dbc.Container(
            html.Div(id="passive-review-table"),
            fluid=True,
            className="dbc"
        ),
        html.Br(),
        html.Br(),
        html.Br()
    ]
)



# NPS page layout with tabs
nps_layout = html.Div(
    [
        dbc.Tabs(
            [                
                dbc.Tab(
                    nps_scores_layout,  # NPS Scores section
                    label="NPS",  # Rename the tab label
                ),
                dbc.Tab(
                    [
                        html.Br(),
                        html.H3("Breakdown of Promoter Data", style={"color": "#a142ff"}),
                        dcc.Graph(
                            id="Promoter Frequency",
                            figure=plot_category_frequency_by_topic("Promoter"),
                        ),
                        html.Br(),
                        promoter_layout,
                    ],
                    label="Promoter",
                ),
                dbc.Tab(
                    [
                        html.Br(),
                        html.H3("Breakdown of Detractor Data", style={"color": "#a142ff"}),
                        dcc.Graph(
                            id="Detractor Frequency",
                            figure=plot_category_frequency_by_topic("Detractor"),
                        ),
                        html.Br(),
                        detractor_layout,
                    ],
                    label="Detractor",
                ),
                dbc.Tab(
                    [
                        html.Br(),
                        html.H3("Breakdown of Passive Data", style={"color": "#a142ff"}),
                        dcc.Graph(
                            id="Passive Frequency",
                            figure=plot_category_frequency_by_topic("Passive"),
                        ),
                        html.Br(),
                        passive_layout,
                    ],
                    label="Passive",
                ),
            ]
        ),
    ]
)

data_issues = pd.read_csv('/app/data/combined_data.csv')
topic_df_issues = pd.read_csv('/app/data/topics_review.csv')


# trends page layout
trends_layout = html.Div(
    [
        html.H1("Trends in reviews",style = {"color":"#a142ff"}),
        html.Div(
            dbc.Alert(
                [
                    html.I(className="bi bi-info-circle-fill me-2"),
                    "Select a topic from the dropdown to view details",
                ],
                color="#4a3b78",
                className="d-flex align-items-center",
            ),
        ),
        dbc.Container(
            dcc.DatePickerRange(
                id='date_picker_range',
                min_date_allowed=get_date_range()[0],
                max_date_allowed=get_date_range()[1],
                initial_visible_month=get_date_range()[0],
                end_date=get_date_range()[1],
                ),
            fluid=True,
            className="dbc"
        ),
        dbc.Container(
            dcc.Dropdown(
                id="topic-dropdown",
                options=[{"label": topic, "value": topic} for topic in topics_issues],
                value=topics_issues[0],
                style = {"width":"50%"},
                placeholder = "Select Topic..."
            ),
            fluid=True,
            className="dbc"
        ),
        dcc.Graph(id="issues-line-chart"),
        html.Br(),
        html.Div(id='issues-table-container'),  # Container for the table
        html.Br(),
        html.Br(),
        html.Br()
    ]
)


# Layout for CSV review rater output
csv_review_rater_layout = html.Div(
    [
        html.H3("CSV Review Rater Output", style={"color": "#9155fa"}),
        dcc.Upload(
            id='upload-data',
            children=html.Button('Click Here to Upload File', style={"font-size": "18px", "background-color": "#2b2b2b",  # Dark background color
                    "color": "#ffffff"}),
                    ),
        html.Div(id="csv-review-output-filename", style={"margin-bottom": "10px"}),
        html.Div(id = "csv-review-output-holder"),
        html.Div(id='parsed-csv-data', style={'display': 'none'}),
        html.Button("Analyse my CSV file", id="csv-review-enter-button"),
        dcc.Loading(
            id="loading-csv-review-output",
            type="default",
            children=[
                html.Div(id="csv-review-output"),
                html.Div("CSV Loading Bar", id="csv-loading-bar")
            ]
        ),
        html.Hr(),  # Break Line
    ]
)

# Layout for textbox review rater
textbox_review_rater_layout = html.Div(
    [
        html.H3("Textbox Review Rater", style={"color": "#9155fa"}),
        dcc.Textarea(
            id="text-input",
            placeholder="Enter review(s) here...",
            rows=5,
            style={
                "width": "100%",
                "background-color": "#2b2b2b",  # Dark background color
                "color": "#ffffff",  # Text color (white)
                "margin-bottom": "10px",  # Add margin bottom
            },
        ),
        html.Button(
            "Analyse my text",
            id="submit-button",
            style={
                "font-size": "18px",
                "background-color": "#2b2b2b",  # Dark background color
                "color": "#ffffff",
                "margin-bottom": "10px",  # Add margin bottom
            },
        ),
        dcc.Loading(
            id="loading-textbox-review-output",
            type="default",
            children=[
                html.Div(id="textbox-review-output"),
                html.Div("Text Loading Bar", id="text-loading-bar")
            ]
        ),
        html.Hr(),  # Break Line
    ]
)

# Combine both layouts into one
review_rater_layout = html.Div(
    [
        html.H1("Review Rater", style={"color": "#9155fa"}),
        html.Div(
            dbc.Alert(
                [
                    html.I(className="bi bi-info-circle-fill me-2"),
                    "Upload your .CSV file or Input new reviews in the textbox below",
                ],
                color="#4a3b78",
                className="d-flex align-items-center",
            )
        ),
        csv_review_rater_layout,
        textbox_review_rater_layout
    ]
)




# credits page layout
credits_layout = html.Div(
    [
        html.H2("Credits",style = {"color":"#9155fa"}),
        html.Div([
            html.H3("Development Team"), 
            html.Ul([
                html.Li("Front End: Aiko Liana Amran | Denise Teh Kai Xin | Low Jia Li Natalie | Ng Yee Gee Kiley"),
                html.Li("Back End: Anthea Ang Qiao En | Chan Wan Xin , Lydia | Lucia Pan Yucheng | Neleh Tok Ying Yun")
            ])
        ]),
        html.Div([
            html.H3("Stakeholders"),
            html.Ul([
                html.Li("Team DKLAN"),
                html.Li("GXS Customer Experience and Business Bevelopment Team"),
                html.Li("DSA3101 Teaching Team")
            ])
        ]),
        html.Div([
            html.H3("Special Thanks"),
            html.Ul([
                html.Li("Opensource community & Model creators")
            ])
        ])
    ]
)

# App layout
app.layout = html.Div([sidebar, content])


# Callbacks
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return nps_layout
    elif pathname == "/trends":
        return trends_layout
    elif pathname == "/review_rater":
        return review_rater_layout
    elif pathname == "/credits":
        return credits_layout
    else:
        return nps_layout



# Define callback to update main graph
@app.callback(
    Output('nps-by-topic-graph', 'figure'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date')
)
def update_main_graph(start_date, end_date):
    # get nps score for all topics
    topic_results = []
    for csv_file in topics_csv:
        data = process_csv(csv_file)
        temp = topic_nps(data, start_date, end_date)
        topic_results.append(temp)

    # store results in a dataframe with topic names specified and sort
    final_df = pd.DataFrame({'Topic': topics, 'NPS': topic_results})
    final_df_sorted = final_df.sort_values(by='NPS', ascending=False)

    # setting the values used to make the axis
    topic_min_nps = final_df_sorted['NPS'].min()
    topic_max_nps = final_df_sorted['NPS'].max()
    topic_cap = max(abs(topic_min_nps), abs(topic_max_nps))

    # making the graph itself
    c_scale = ['red', 'orange', 'green']
    fig = px.bar(final_df_sorted, x='Topic', y='NPS', title='NPS Score by Topic', color='NPS',
                color_continuous_scale=c_scale, color_continuous_midpoint=0)
    fig.update_yaxes(range=[-topic_cap, topic_cap])
    return fig

# Define callback to update drill-down graph
@app.callback(
    Output('drill-down-graph', 'figure'),
    Output('drill-down-graph', 'style'),  # Add an output for style to hide the graph
    Output('drill-down-container', 'children'),  # Add an output for the container
    Input('nps-by-topic-graph', 'clickData'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date')
)
def update_drill_down_graph(clickData, start_date, end_date):
    if clickData is None:
        # If no data point is clicked, return an empty figure and hide the graph
        return {}, {'display': 'none'}, html.Br()
    
    # Get the clicked topic
    topic = clickData['points'][0]['x']

    # get the df of issue nps scores only for the topic clicked
    file_path = f"{topic}.csv"
    data = process_csv(file_path)
    issue_results = issue_nps(data, start_date, end_date)

    # sort by nps 
    issue_results_sorted = issue_results.sort_values(by='NPS', ascending=False)

    # setting the values used to make the axis
    issue_min_nps = issue_results_sorted['NPS'].min()
    issue_max_nps = issue_results_sorted['NPS'].max()
    issue_cap = max(abs(issue_min_nps), abs(issue_max_nps))

    # making the graph itself
    c_scale = ['red', 'orange', 'green']
    fig = px.bar(issue_results_sorted, x='Issue', y='NPS', title=f'NPS for {topic} Subtopic', color='NPS',
                 color_continuous_scale=c_scale, color_continuous_midpoint=0)
    fig.update_yaxes(range=[-issue_cap, issue_cap])
    
    return fig, {'display': 'block'}, html.Div()  # Return the graph and show it, and an empty Div for the container

# Callback to update the review table based on dropdown selection
@app.callback(
    Output('promoter-review-table', 'children'),
    [Input('promoter-topic-dropdown', 'value')]
)
def update_promoter_review_table(selected_topic):
    return table_reviews_category("Promoter", selected_topic)

# Callback to update the review table based on dropdown selection
@app.callback(
    Output('detractor-review-table', 'children'),
    [Input('detractor-topic-dropdown', 'value')]
)
def update_detractor_review_table(selected_topic):
    return table_reviews_category("Detractor", selected_topic)

# Callback to update the review table based on dropdown selection
@app.callback(
    Output('passive-review-table', 'children'),
    [Input('passive-topic-dropdown', 'value')]
)
def update_passive_review_table(selected_topic):
    return table_reviews_category("Passive", selected_topic)



def update_date_range(fig, start_date, end_date):
    if start_date is None:
        start_date = get_date_range()[0]
    if end_date is None:
        end_date = get_date_range()[1]
    fig.update_xaxes(range=[start_date, end_date])
    return fig

def serialize_figure(fig):
    # Convert Period objects to strings before serializing
    fig_dict = fig.to_dict()
    for data_entry in fig_dict['data']:
        if 'x' in data_entry:
            data_entry['x'] = [str(period) for period in data_entry['x']]
        if 'y' in data_entry:
            if isinstance(data_entry['y'], np.ndarray):
                data_entry['y'] = data_entry['y'].tolist()
    return fig_dict

@app.callback(
    [Output("issues-line-chart", "figure"),
     Output('issues-table-container', 'children')],  # Add output for the table
    [Input("topic-dropdown", "value"),
     Input('date_picker_range', 'start_date'),
     Input('date_picker_range', 'end_date'),]
)
def update_issues_page(topic, start_date, end_date):
    if not topic:  # Check if topic is None or empty string
        # Return default graph and an empty table
        default_fig = plot_default_graph()
        update_date_range(default_fig, start_date, end_date)
        return serialize_figure(default_fig), html.Div()
    else:
        # Define a dictionary mapping topics to their respective DataFrames
        topic_to_df = {
            'App Responsiveness': app_responsiveness,
            'Competition': competition,
            'Credit card usage': credit_card,
            'Customer Services': customer_service,
            'Customer Trust': customer_trust,
            'Login & Account Setup': login_account,
            'Money Growth (Interest Rates)': money_growth,
            'Safety': safety,
            'Service Products': service_products,
            'User Interface': user_interface
        }

        # Get the DataFrame for the selected topic
        df = topic_to_df[topic]
        cleaned_df = preprocess(data_issues, df)
        # Call plot_top_n_issues_time_series function to generate the figure
        # Get the DataFrame of related issues and solutions for the selected topic
        related_issues_df = select_related_solutions(topic, solutions_df)
        
        # Generate the graph
        fig = plot_top_n_issues_time_series(cleaned_df)
        update_date_range(fig, start_date, end_date)
        
        # Generate the table
        related_issues_df = related_issues_df.rename(columns={'Issue': 'Subtopic', 'Solution': 'Insight'})
        table = dbc.Container([
                    html.Div(
                        dbc.Alert(
                            [
                                html.I(className="bi bi-info-circle-fill me-2"),
                                "To filter: Key in the words you want and press Enter.  To remove filter: Clear text and press Enter"
                            ],
                            color="#4a3b78",
                            className="d-flex align-items-center",
                        ),
                    ),
                    html.Br(),
                    dash_table.DataTable(
                    id='issues-table',
                    columns=[{"name": i, "id": i} for i in related_issues_df.columns],
                    data=related_issues_df.to_dict('records'),
                    style_table={'borderRadius': '15px'},  # Dark background color
                    style_header={'backgroundColor': '#6a05ed', 'fontWeight': 'bold', 'color': '#fafaf9'},  # Purple header text
                    style_cell={'backgroundColor': '#303030', 'textAlign': 'left', 'padding': '5px', 'color': '#fafaf9'},  # Light text color
                    filter_action='native',  # Enable filtering
                    filter_options={"placeholder_text": "Enter Filter Word Here..."},
                    sort_action='native',  # Enable native sorting
                    sort_by=[{'column_id': 'subtopic', 'direction': 'asc'}]  # Default sorting by 'Name' column in ascending order
                    )],
                fluid=True,
                className="dbc"
        )

        
        return serialize_figure(fig), table  # Return both the graph and the table



# Define the parse_contents function to handle uploaded CSV files
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    # Concatenate text content of all columns into one long string
    text_content = ' '.join(df.applymap(str).values.flatten())

    # Return the text content along with the filename
    return html.Div([
        html.H5(filename),
        html.Hr(),  # horizontal line
    ])


# Callback to handle updating output data upload for CSV files
@app.callback(
    [Output('csv-review-output-holder', 'children'),
     Output("parsed-csv-data", "data")],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified')]
)
def update_csv_output(list_of_contents, list_of_names, list_of_dates):
    try:
        if list_of_contents is not None:
            # Perform processing of CSV content here
            parsed_data = [
                parse_contents(c, n, d) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]
            return parsed_data, parsed_data
        else:
            return None, ""
    except Exception as e:
        print("Error occurred while updating CSV output:")
        print(e)
        return html.Div(["Error occurred while processing the CSV file."]), ""

# Callback to handle updating output data for CSV input
@app.callback(
    Output("csv-review-output", "children"),
    Output("csv-loading-bar", "children"),
    [Input("csv-review-enter-button", "n_clicks")],
    [State('parsed-csv-data', "data")]
)
def update_csv_output(n_clicks, parsed_data):
    if n_clicks is not None and parsed_data:
        # Simulate processing delay
        time.sleep(1)
        
        # Code for NPS rater output using text input
        nps_score_output = nps_score(parsed_data)
        nps_category_output = nps_cat(parsed_data)
        nps_review_output = review_analysis(parsed_data)
        
        # Split the review analysis into paragraphs
        paragraphs = nps_review_output.split('\n\n')
        
        # Create a list of HTML div elements for each paragraph
        review_divs = [html.Div(paragraph, style={'margin-top': '20px', 'fontSize': '14px'}) for paragraph in paragraphs]
        
        # Display NPS score and category with review analysis
        nps_output_content = [
            html.Div(f"Net Promoter Score: {nps_score_output}/10", style={'fontSize': '16px', 'font-weight': 'bold', 'margin-top': '20px'}),
            html.Div(f"NPS Category: {nps_category_output}", style={'fontSize': '16px', 'font-weight': 'bold', 'margin-top': '10px'}),
            html.Div("Summary of review:", style={'fontSize': '16px', 'font-weight': 'bold', 'margin-bottom': '20px'}),
            *review_divs,
        ]
        return nps_output_content, ""
    
    # If not processing, return empty content
    return "", ""


# Callback to handle updating output data for CSV input
@app.callback(
    Output("textbox-review-output", "children"),
    Output("text-loading-bar", "children"),
    [Input("submit-button", "n_clicks")],
    [State("text-input", "value")]
)
def update_textbox_output(n_clicks, sentence):
    if n_clicks is not None:
        # Simulate processing delay
        time.sleep(1)
        
        # Code for NPS rater output using text input
        nps_score_output = nps_score(sentence)
        nps_category_output = nps_cat(sentence)
        nps_review_output = review_analysis(sentence)

        # Split the review analysis into paragraphs
        paragraphs = nps_review_output.split('\n\n')
        
        # Create a list of HTML div elements for each paragraph
        review_divs = [html.Div(paragraph, style={'margin-top': '20px', 'fontSize': '14px'}) for paragraph in paragraphs]
        
        # Display NPS score and category with review analysis
        nps_output_content = [
            html.Div(f"Net Promoter Score: {nps_score_output}/10", style={'fontSize': '16px', 'font-weight': 'bold', 'margin-top': '20px'}),
            html.Div(f"NPS Category: {nps_category_output}", style={'fontSize': '16px', 'font-weight': 'bold', 'margin-top': '10px'}),
            html.Div("Summary of review:", style={'fontSize': '16px', 'font-weight': 'bold', 'margin-bottom': '20px'}),
            *review_divs,
        ]
        
        return nps_output_content, ""
    
    # If not processing, return empty content
    return "", ""


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)
