# Import necessary modules
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


# Set Directory (For Local version only)
os.chdir("C:/Users/rhyde/SentimentPro/Data")

# Read Data

## Dashboard Page:
topics_df = pd.read_csv('TopicsofReviews.csv')
nps_df = pd.read_csv('nps_df.csv')
score_df = pd.read_csv('score_df.csv')
date_df = pd.read_csv('combined_data.csv')

## Issues Page:
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


# Functions

## Dashboard Code:

### Wrap function to read and edit CSV files
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

### Topic NPS Scorer
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

### Issue NPS Scorer
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


# list out csv file names
topics_csv = ['App Responsiveness.csv', 'Competition.csv', 'Credit card usage.csv', 'Customer Services.csv', 'Customer trust.csv', 
              'Login & Account Setup.csv', 'Money Growth (Interest Rates).csv', 'Safety.csv', 'Services & Products.csv', 'User Interface.csv']

# list out topic names
topics = ['App Responsiveness', 'Competition', 'Credit card usage', 'Customer Services', 'Customer trust', 'Login & Account Setup', 
          'Money Growth (Interest Rates)', 'Safety', 'Services & Products', 'User Interface']



# Issues Page
data = pd.read_csv('combined_data.csv')
topic_df = pd.read_csv('topics_review.csv')
def plot_default_graph():
    # Merge the dataframes
    all_data = data.merge(topic_df, on='review', how='inner')

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
    all_data = data.merge(topic_df, on='review', how='inner')

    # Clean the date
    all_data['date_clean'] = pd.to_datetime(all_data['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    all_data['date_clean'] = all_data['date_clean'].combine_first(pd.to_datetime(all_data['date'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce'))
    all_data['date_clean'] = all_data['date_clean'].astype(str).str[:10]

    min_date = all_data.date_clean.min()
    max_date = all_data.date_clean.max()
    return [min_date, max_date]

# Test
topics = ['', 'App Responsiveness', 'Competition', 'Credit Card Usage', 'Customer Services', 'Customer Trust',
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



