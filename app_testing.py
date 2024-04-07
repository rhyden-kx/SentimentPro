# Import necessary libraries
import os
import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import matplotlib.pyplot as plt 
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from textblob import TextBlob
import re
import time
import random


#Change Directory:
os.chdir('C:/Users/rhyde/SentimentPro/Data')

# Read CSV files
nps_df = pd.read_csv('nps_df.csv')
topics_df = pd.read_csv('TopicsofReviews.csv')
score_df = pd.read_csv('score_df.csv')
app_responsiveness = pd.read_csv('App Responsiveness.csv')
competition = pd.read_csv('Competition.csv')
credit_card = pd.read_csv('Credit card usage.csv')
customer_service = pd.read_csv('Customer Services.csv')
customer_trust = pd.read_csv('Customer trust.csv')
login_account = pd.read_csv('Login & Account Setup.csv')
money_growth = pd.read_csv('Money Growth (Interest Rates).csv')
safety = pd.read_csv('Safety.csv')
service_products = pd.read_csv('Services & Products.csv')
user_interface = pd.read_csv('User Interface.csv')
data = pd.read_csv('combined_data.csv')

#Time Series module
import Time_Series_Processing

clean_app_responsiveness = Time_Series_Processing.preprocess(data, app_responsiveness)
clean_app_responsiveness_graph = Time_Series_Processing.plot_top_n_issues_time_series(clean_app_responsiveness, top_n=3)


# Set up the Dash app
app = dash.Dash(__name__)

# Define the layout of your Dash app
app.layout = html.Div([
    dcc.Graph(id='top-issues-time-series')
])

# Define callback to update the graph
@app.callback(
    Output('top-issues-time-series', 'figure'),
    [Input('dropdown', 'value')]  # Assuming you have a dropdown for selecting top N issues
)
def update_graph(selected_value):
    # Plot top N issues over time
    fig = clean_app_responsiveness_graph
    return fig


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
