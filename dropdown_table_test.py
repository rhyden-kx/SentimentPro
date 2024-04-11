import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import random
import plotly.graph_objs as go

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

# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Issues page layout
issues_layout = html.Div(
    [
        html.H1("Issues Faced"),
        html.H3("Our Solutions"),
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

# App layout
app.layout = issues_layout

# Callback to update line chart and solutions table based on selected topic
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

if __name__ == "__main__":
    app.run_server(debug=True)
