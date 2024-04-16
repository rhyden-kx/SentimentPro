import os
import json
import random
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
from datetime import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Set Directory (For Local version only)
os.chdir("C:/Users/rhyde/SentimentPro/Data")

# Read Data

## Dashboard Page:
topics_df = pd.read_csv('TopicsofReviews.csv')
nps_df = pd.read_csv('nps_df.csv')
score_df = pd.read_csv('score_df.csv')
date_df = pd.read_csv('combined_data.csv')

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Wrap function to read and edit CSV files
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

app_responsiveness = pd.read_csv('App Responsiveness.csv')
competition = pd.read_csv('Competition.csv')
credit_card = pd.read_csv('Credit card.csv')
customer_service = pd.read_csv('Customer Services.csv')
customer_trust = pd.read_csv('Customer trust.csv')
login_account = pd.read_csv('Login & Account Setup.csv')
money_growth = pd.read_csv('Money Growth (Interest Rates).csv')
safety = pd.read_csv('Safety.csv')
service_products = pd.read_csv('Services & Products.csv')
user_interface = pd.read_csv('User Interface.csv')
data = pd.read_csv('combined_data.csv')

# Test
topics_issues = ['', 'App Responsiveness', 'Competition', 'Credit Card Usage', 'Customer Services', 'Customer Trust',
          'Login & Account Setup', 'Money Growth (Interest Rates)', 'Safety', 'Service Products', 'User Interface']

datasets = {
    'App Responsiveness': app_responsiveness,
    'Competition': competition,
    'Credit Card Usage': credit_card,
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

    # Create a time series line plot
    fig = px.line(monthly_data_top, x='month_year', y='count', color='issue', title=f'Number of Reviews by Top 5 Issues Over Time')
    
    # Add markers to the lines
    for trace in fig.data:
        trace.update(mode='lines+markers')
    fig.update_xaxes(title_text='Month')
    fig.update_yaxes(title_text='Number of Reviews')
    fig.update_yaxes(fixedrange=True)
    return fig





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

# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Sidebar
sidebar = html.Div(
    [
        html.H2("SentimentPro", className="display-4", style={"font-size": "2rem"}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Dashboard", href="/", active="exact"),
                dbc.NavLink("Issues", href="/issues", active="exact"),
                dbc.NavLink("NPS Rater", href="/nps_rater", active="exact"),
                dbc.NavLink("Information", href="/information", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar",
    className="bg-light",
    style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "16rem", "padding": "2rem", "transition": "0.5s"},
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
        html.H1("Net Promoter Score (NPS) Analysis"),
        html.P("Click on a bar in the graph below to see NPS scores by issues for that topic."),
        dcc.DatePickerRange(
            id='date-range-picker',
            start_date='2022-08-01',
            end_date=dt.now().date(),
            display_format='YYYY-MM-DD'
        ),
        dcc.Graph(id='nps-by-topic-graph'),
        html.Div(id='drill-down-container'),
        dcc.Graph(id='drill-down-graph')
    ]
)

# Dashboard page layout with tabs
dashboard_layout = html.Div(
    [
        html.H1("Dashboard"),
        dbc.Tabs(
            [
                dbc.Tab(
                    nps_scores_layout,  # NPS Scores section
                    label="NPS Scores",  # Rename the tab label
                ),
                dbc.Tab(
                    [
                        html.H3("Good"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        id="good-line-chart",
                                        figure={
                                            "data": [go.Scatter(x=[1, 2, 3], y=[4, 1, 2], mode="lines")],
                                            "layout": go.Layout(title="Good Line Chart"),
                                        },
                                    ),
                                    width=6,
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        id="good-bar-chart",
                                        figure={
                                            "data": [go.Bar(x=[1, 2, 3], y=[4, 1, 2])],
                                            "layout": go.Layout(title="Good Bar Chart"),
                                        },
                                    ),
                                    width=6,
                                ),
                            ]
                        ),
                    ],
                    label="Good",
                ),
                dbc.Tab(
                    [
                        html.H3("Bad"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        id="bad-line-chart",
                                        figure={
                                            "data": [go.Scatter(x=[1, 2, 3], y=[4, 1, 2], mode="lines")],
                                            "layout": go.Layout(title="Bad Line Chart"),
                                        },
                                    ),
                                    width=6,
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        id="bad-bar-chart",
                                        figure={
                                            "data": [go.Bar(x=[1, 2, 3], y=[4, 1, 2])],
                                            "layout": go.Layout(title="Bad Bar Chart"),
                                        },
                                    ),
                                    width=6,
                                ),
                            ]
                        ),
                    ],
                    label="Bad",
                ),
                dbc.Tab(
                    [
                        html.H3("Neutral"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        id="neutral-line-chart",
                                        figure={
                                            "data": [go.Scatter(x=[1, 2, 3], y=[4, 1, 2], mode="lines")],
                                            "layout": go.Layout(title="Neutral Line Chart"),
                                        },
                                    ),
                                    width=6,
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        id="neutral-bar-chart",
                                        figure={
                                            "data": [go.Bar(x=[1, 2, 3], y=[4, 1, 2])],
                                            "layout": go.Layout(title="Neutral Bar Chart"),
                                        },
                                    ),
                                    width=6,
                                ),
                            ]
                        ),
                    ],
                    label="Neutral",
                ),
            ]
        ),
    ]
)

data_issues = pd.read_csv('combined_data.csv')
topic_df_issues = pd.read_csv('topics_review.csv')
# Issues page layout
issues_layout = html.Div(
    [
        html.H1("Issues Faced"),
        html.H3("Select an issue to view details"),
        html.Br(),
        dcc.DatePickerRange(
            id='date_picker_range',
            min_date_allowed=get_date_range()[0],
            max_date_allowed=get_date_range()[1],
            initial_visible_month=get_date_range()[0],
            end_date=get_date_range()[1]),
        html.Br(),
        dcc.Dropdown(
            id="topic-dropdown",
            options=[{"label": topic, "value": topic} for topic in topics_issues],
            value=topics_issues[0],
        ),
        dcc.Graph(id="issues-line-chart"),
    ]
)

# NPS Rater page layout
nps_rater_layout = html.Div(
    [
        html.H1("NPS Rater"),
        html.H3("Input new reviews here"),
        dcc.Textarea(id="text-input", placeholder="Enter text here...", rows=5, style={"width": "100%"}),
        html.Br(),
        html.Button("Enter", id="submit-button"),
        html.Br(),
        html.Div(id="output-text"),
    ]
)

# Information page layout
information_layout = html.Div(
    [
        html.H2("Credits"),
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
        return dashboard_layout
    elif pathname == "/issues":
        return issues_layout
    elif pathname == "/nps_rater":
        return nps_rater_layout
    elif pathname == "/information":
        return information_layout
    else:
        return dashboard_layout

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

    # store results in a datframe with topic names specified and sort
    final_df = pd.DataFrame({'Topic': topics, 'NPS': topic_results})
    final_df_sorted = final_df.sort_values(by='NPS', ascending=False)

    # setting the values used to make the axis
    topic_min_nps = final_df_sorted['NPS'].min()
    topic_max_nps = final_df_sorted['NPS'].max()
    topic_cap = max(abs(topic_min_nps), abs(topic_max_nps))

    # making the graph itself
    c_scale = ['red', 'orange', 'green']
    fig = px.bar(final_df_sorted, x='Topic', y='NPS', title='NPS Score by Topic', color='NPS',
                color_continuous_scale = c_scale, color_continuous_midpoint=0)
    fig.update_yaxes(range=[-topic_cap, topic_cap])
    return fig

# Define callback to update drill-down graph
@app.callback(
    Output('drill-down-graph', 'figure'),
    Input('nps-by-topic-graph', 'clickData'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date')
)

def update_drill_down_graph(clickData, start_date, end_date):
    if clickData is None:
        # If no data point is clicked, return an empty figure
        return {}
    
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
    fig = px.bar(issue_results_sorted, x='Issue', y='NPS', title=f'NPS Score for {topic} Issues', color='NPS',
                 color_continuous_scale=c_scale, color_continuous_midpoint=0)
    fig.update_yaxes(range=[-issue_cap, issue_cap])
    
    return fig

def update_date_range(fig, start_date, end_date):
    if start_date is None:
        start_date = get_date_range()[0]
    if end_date is None:
        end_date = get_date_range()[1]
    fig.update_xaxes(range=[start_date,end_date])
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
    Output("issues-line-chart", "figure"),
    [Input("topic-dropdown", "value"),
    Input('date_picker_range', 'start_date'),
    Input('date_picker_range', 'end_date')]
)
def update_issues_page(topic, start_date, end_date):
    if not topic:  # Check if topic is None or empty string
        # Return default graph
        default_fig = plot_default_graph()
        update_date_range(default_fig, start_date, end_date)
        return serialize_figure(default_fig)  # Serialize Figure manually
    else:
        # Define a dictionary mapping topics to their respective DataFrames
        topic_to_df = {
            'App Responsiveness': app_responsiveness,
            'Competition': competition,
            'Credit Card Usage': credit_card,
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
        fig = plot_top_n_issues_time_series(cleaned_df)
        update_date_range(fig, start_date, end_date)
        return serialize_figure(fig)  # Serialize Figure manually
    









@app.callback(
    [Output("output-text", "children")],
    [Input("submit-button", "n_clicks")],
    [State("text-input", "value")]
)
def update_output(n_clicks, sentence):
    if n_clicks is not None and sentence is not None:
        # Code for NPS rater output
        nps_score_output = nps_score(sentence)
        nps_category_output = nps_cat(sentence)
        nps_review_output = review_analysis(sentence)
        
        # Split the review analysis into paragraphs
        paragraphs = nps_review_output.split('\n\n')
        
        # Create a list of HTML div elements for each paragraph
        review_divs = [html.Div(paragraph, style={'margin-top': '20px','fontSize': '14px',}) for paragraph in paragraphs]
        
        # Combine the review divs with NPS score and category
        nps_output_content = [
            html.Div("Summary of review:", style={'fontSize': '16px', 'fontFamily': 'Courier New, serif','font-weight': 'bold', 'margin-bottom': '10px', 'color': '#444454'}),
            *review_divs,
            html.Div(f"NPS Score: {nps_score_output}/10", style={'fontSize': '16px','fontFamily': 'Courier New, serif','font-weight': 'bold', 'margin-top': '20px', 'color': '#444454'}),
            html.Div(f"Category: {nps_category_output}", style={'fontSize': '16px','fontFamily': 'Courier New, serif','font-weight': 'bold', 'margin-top': '10px', 'color': '#444454'})
        ]
        
        return [nps_output_content]
    else:
        return ['']



if __name__ == "__main__":
    app.run_server(debug=True)
