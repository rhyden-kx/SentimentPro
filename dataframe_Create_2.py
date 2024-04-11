import os
import pandas as pd

os.chdir("C:/Users/rhyde/SentimentPro/Data")

import pandas as pd

# Read the score_df CSV file
score_df = pd.read_csv("score_df.csv")

# Split rows with multiple Topic Numbers into separate rows
score_df['Topic_Number'] = score_df['Topic_Number'].str.split(',')
score_df = score_df.explode('Topic_Number')

# Map the Topic Names based on the assigned Topic Numbers
topic_mapping = {
    '0': 'App Responsiveness',
    '1': 'Money Growth (Interest Rates)',
    '2': 'Customer Services',
    '3': 'Services & Products',
    '4': 'User Interface',
    '5': 'Credit card usage',
    '6': 'Login & Account Setup',
    '7': 'Competition',
    '8': 'Safety',
    '9': 'Customer trust'
}
score_df['Topic_Name'] = score_df['Topic_Number'].map(topic_mapping)

# Save the cleaned DataFrame to a new CSV file
score_df.to_csv("cleaned_score_df_2.csv", index=False)
