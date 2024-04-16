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


# Issues page layout
issues_layout = html.Div(
    [
        html.H1("Issues Faced"),
        html.H3("Select an issue to view details"),
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
