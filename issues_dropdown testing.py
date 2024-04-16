import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import random
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os
import json

# Set Directory (For Local version only)
os.chdir("C:/Users/rhyde/SentimentPro/Data")

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


# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
            options=[{"label": topic, "value": topic} for topic in topics],
            value=topics[0],
        ),
        dcc.Graph(id="issues-line-chart"),
    ]
)

# App layout
app.layout = issues_layout

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
    

    
if __name__ == "__main__":
    app.run_server(debug=True)