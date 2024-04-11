import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Output, Input
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime as dt
import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

os.chdir("C:/Users/rhyde/SentimentPro/Data")

topics_df = pd.read_csv('TopicsofReviews.csv')
nps_df = pd.read_csv('nps_df.csv')
score_df = pd.read_csv('score_df.csv')
date_df = pd.read_csv('combined_data.csv')


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

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Net Promoter Score (NPS) Analysis"),
    html.P("Click on a bar in the graph below to see NPS scores by issues for that topic."),
    dcc.DatePickerRange(
        id='date-range-picker',
        start_date='2022-08-01',
        end_date=dt.now().date(),
        display_format='YYYY-MM-DD'
    ),
    dcc.Graph(id='nps-by-topic-graph'),
    html.Div(id='drill-down-container')
])

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
                color_continuous_scale=c_scale, color_continuous_midpoint=0)
    fig.update_yaxes(range=[-topic_cap, topic_cap])
    return fig

# Define callback to update drill-down graph
@app.callback(
    Output('drill-down-container', 'children'),
    Input('nps-by-topic-graph', 'clickData'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date')
)
def update_drill_down_graph(clickData, start_date, end_date):
    if clickData is None:
        return html.Div()
    
    topic = clickData['points'][0]['x']
    file_path = f"{topic}.csv"
    data = process_csv(file_path)
    issue_results = issue_nps(data, start_date, end_date)
    issue_results_sorted = issue_results.sort_values(by='NPS', ascending=False)
    issue_min_nps = issue_results_sorted['NPS'].min()
    issue_max_nps = issue_results_sorted['NPS'].max()
    issue_cap = max(abs(issue_min_nps), abs(issue_max_nps))
    c_scale = ['red', 'orange', 'green']
    fig = px.bar(issue_results_sorted, x='Issue', y='NPS', title=f'NPS Score for {topic} Issues', color='NPS',
                 color_continuous_scale=c_scale, color_continuous_midpoint=0)
    fig.update_yaxes(range=[-issue_cap, issue_cap])
    return dcc.Graph(figure=fig)


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
