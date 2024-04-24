# Define the parse_contents function to handle uploaded files
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
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
    return (text_content, filename)

# Callback to handle CSV file rating
@app.callback(
    Output('csv-review-output-holder', 'children'),
    Input('csv-enter-button', 'n_clicks'),
    State('output-data-upload', 'children')
)
def rate_reviews_from_csv(n_clicks, uploaded_contents):
    if n_clicks and uploaded_contents:
        # Iterate through each uploaded file
        for content, filename in uploaded_contents:
            # Perform NPS rating on the uploaded text
            nps_score_output = nps_score(content)
            nps_category_output = nps_cat(content)
            nps_review_output = review_analysis(content)
            
            # Split the review analysis into paragraphs
            paragraphs = nps_review_output.split('\n\n')
            
            # Create a list of HTML div elements for each paragraph
            review_divs = [
                html.Div(
                    paragraph,
                    style={'margin-top': '20px', 'fontSize': '14px'}
                ) for paragraph in paragraphs
            ]
            
            # Combine the review divs with NPS score and category
            nps_output_content = [
                html.Div(
                    f"Net Promoter Score: {nps_score_output}/10",
                    style={'fontSize': '16px', 'font-weight': 'bold', 'margin-top': '20px'}
                ),
                html.Div(
                    f"NPS Category: {nps_category_output}",
                    style={'fontSize': '16px', 'font-weight': 'bold', 'margin-top': '10px'}
                ),
                html.Div(
                    "Summary of review:",
                    style={'fontSize': '16px', 'font-weight': 'bold', 'margin-bottom': '20px'}
                ),
                *review_divs,
            ]
            
            return nps_output_content
    return html.Div()

# Callback to handle text input rating
@app.callback(
    Output("textbox-review-output", "children"),
    [Input("submit-button", "n_clicks")],
    [State("text-input", "value")]
)
def update_output(n_clicks, sentence):
    if n_clicks is not None and sentence:
        # Code for NPS rater output
        nps_score_output = nps_score(sentence)
        nps_category_output = nps_cat(sentence)
        nps_review_output = review_analysis(sentence)
        
        # Split the review analysis into paragraphs
        paragraphs = nps_review_output.split('\n\n')
        
        # Create a list of HTML div elements for each paragraph
        review_divs = [html.Div(paragraph, style={'margin-top': '20px', 'fontSize': '14px'}) for paragraph in paragraphs]
        
        # Combine the review divs with NPS score and category
        nps_output_content = [
            html.Div("Summary of review:", style={'fontSize': '16px', 'fontFamily': 'Courier New, serif', 'font-weight': 'bold', 'margin-bottom': '10px'}),
            *review_divs,
            html.Div(f"NPS Score: {nps_score_output}/10", style={'fontSize': '16px', 'fontFamily': 'Courier New, serif', 'font-weight': 'bold', 'margin-top': '20px'}),
            html.Div(f"Category: {nps_category_output}", style={'fontSize': '16px', 'fontFamily': 'Courier New, serif', 'font-weight': 'bold', 'margin-top': '10px'})
        ]
        
        return nps_output_content

    # If not processing, return empty content
    return ""
