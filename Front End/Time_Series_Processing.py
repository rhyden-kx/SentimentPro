import dash
import plotly.express as px
import pandas as pd
import random



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


def plot_top_n_issues_time_series(merged_data, top_n=5):
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
    top_issues = issue_totals.head(top_n).index

    # Filter monthly_data for top n issues
    monthly_data_topn = monthly_data[monthly_data['issue'].isin(top_issues)]

    # Convert 'month_year' to string format
    monthly_data_topn.loc[:, 'month_year'] = monthly_data_topn['month_year'].astype(str)

    # Create a time series line plot
    fig = px.line(monthly_data_topn, x='month_year', y='count', color='issue', title=f'Number of Reviews by Top {top_n} Issues Over Time')
    
    # Add markers to the lines
    for trace in fig.data:
        trace.update(mode='lines+markers')
    fig.update_xaxes(title_text='Month')
    fig.update_yaxes(title_text='Number of Reviews')
    fig.update_yaxes(fixedrange=True)
    fig.show()
