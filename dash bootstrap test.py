import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import random

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



#NPS Scoring
#VADER

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

#nps_scoring
#vader function
#2. Output should be: nps_indiv, nps_category, topic (?) from the score_df ?

def nps_score(review) :
    vs = analyzer.polarity_scores(review)
    pos_score = vs['pos']
    neg_score = vs['neg']
    neu_score = vs['neu']
    comp_score = vs['compound']
    nps_indv = -1
    #mapping
    if -1 <= vs['compound'] <= -9/11:
        nps_indiv = 0
    elif -9/11 < vs['compound'] <= -7/11:
        nps_indiv = 1
    elif -7/11 < vs['compound'] <= -5/11:
        nps_indiv = 2
    elif -5/11 < vs['compound'] <= -3/11:
        nps_indiv = 3
    elif -3/11 < vs['compound'] <= -1/11:
        nps_indiv = 4
    elif -1/11 < vs['compound'] <= 1/11:
        nps_indiv = 5
    elif 1/11 < vs['compound'] <= 3/11:
        nps_indiv = 6
    elif 3/11 < vs['compound'] <= 5/11:
        nps_indiv = 7
    elif 5/11 < vs['compound'] <= 7/11:
        nps_indiv = 8
    elif 7/11 < vs['compound'] <= 9/11:
        nps_indiv = 9
    else:
        nps_indiv = 10
    return nps_indiv


#nps category
def nps_cat(review) :
    nps_indiv = nps_score(review)
    cat = ""
    if nps_indiv >= 9:  # Promoters
        cat = 'Promoter'
    elif nps_indiv >= 7:  # Passives
        cat = 'Passive'
    else:  # Detractors
        cat = 'Detractor'
    return cat


import os


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




# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Sidebar
sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Dashboard", href="/", active="exact"),
                dbc.NavLink("NPS Scores", href="/nps", active="exact"),
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

# Dashboard page layout with tabs
dashboard_layout = html.Div(
    [
        html.H1("Dashboard"),
        dbc.Tabs(
            [
                dbc.Tab(
                    [
                        html.H3("Overall"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        id="overall-line-chart",
                                        figure={
                                            "data": [go.Scatter(x=[1, 2, 3], y=[4, 1, 2], mode="lines")],
                                            "layout": go.Layout(title="Overall Line Chart"),
                                        },
                                    ),
                                    width=6,
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        id="overall-bar-chart",
                                        figure={
                                            "data": [go.Bar(x=[1, 2, 3], y=[4, 1, 2])],
                                            "layout": go.Layout(title="Overall Bar Chart"),
                                        },
                                    ),
                                    width=6,
                                ),
                            ]
                        ),
                    ],
                    label="Overall",
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

# NPS Scores page layout
# NPS Scores page layout
nps_scores_layout = html.Div(
    [
        html.H1("NPS Scores"),
        html.H3("NPS Trends Over Time"),
        dcc.Graph(
            id="line-chart-nps",
            figure={
                "data": [
                    go.Scatter(x=[1, 2, 3], y=[4, 1, 2], mode="lines", name="Product A"),
                    go.Scatter(x=[1, 2, 3], y=[3, 2, 1], mode="lines", name="Product B"),
                    go.Scatter(x=[1, 2, 3], y=[2, 3, 4], mode="lines", name="Product C"),
                ],
                "layout": go.Layout(title="NPS Trends Over Time"),
            },
        ),
        html.H3("NPS Distribution"),
        dcc.Graph(
            id="bar-chart-nps",
            figure={
                "data": [
                    go.Bar(x=["Promoter", "Passive", "Detractor"], y=[30, 50, 20], name="Product A"),
                    go.Bar(x=["Promoter", "Passive", "Detractor"], y=[40, 30, 30], name="Product B"),
                    go.Bar(x=["Promoter", "Passive", "Detractor"], y=[20, 60, 20], name="Product C"),
                ],
                "layout": go.Layout(title="NPS Distribution"),
            },
        ),
    ]
)
# Issues page layout
issues_layout = html.Div(
    [
        html.H1("Issues Faces"),
        html.H3("Our solutions"),
        html.Br(),
        dcc.Dropdown(
            id="topic-dropdown",
            options=[{"label": topic, "value": topic} for topic in topics],
            value=topics[0],
        ),
        dcc.Graph(id="issues-line-chart"),
        html.Br(),
        html.Table(id="solutions-table"),
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
    elif pathname == "/nps":
        return nps_scores_layout
    elif pathname == "/issues":
        return issues_layout
    elif pathname == "/nps_rater":
        return nps_rater_layout
    elif pathname == "/information":
        return information_layout
    else:
        return dashboard_layout


@app.callback(
    [Output("issues-line-chart", "figure"), Output("solutions-table", "children")],
    [Input("topic-dropdown", "value")],
)
def update_issues_page(topic):
    random_issues = random.choice(list(issues.values()))
    random_solutions = random.choice(list(solutions.values()))
    line_chart_fig = {
        "data": [go.Scatter(x=[1, 2, 3, 4, 5], y=random_issues, mode="lines")],
        "layout": go.Layout(title=f"Issues for {topic}"),
    }
    table_rows = [
        html.Tr([html.Td(issue), html.Td(solution)]) for issue, solution in zip(random_issues, random_solutions)
    ]
    return line_chart_fig, [html.Table([html.Th("Issue"), html.Th("Solution")]), html.Tbody(table_rows)]


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
