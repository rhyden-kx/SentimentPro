import os
import pandas as pd

os.chdir("C:/Users/rhyde/SentimentPro/Data")

# Read the CSV files
topics_df = pd.read_csv("TopicsofReviews.csv")
score_df = pd.read_csv("score_df.csv")
combined_data_df = pd.read_csv("combined_data.csv")
topic_issue_df = pd.read_csv("topic issue combined.csv")

# Convert date columns to datetime format
combined_data_df['date'] = pd.to_datetime(combined_data_df['date'], format='ISO8601', utc=True)
combined_data_df['devresp_time'] = pd.to_datetime(combined_data_df['devresp_time'], format='ISO8601', utc=True)

# Merge topics_df with score_df
merged_df = pd.merge(topics_df, score_df, on=['clean_review', 'Topic_Number', 'Topic_Name'], how='left')

# Merge combined_data_df with merged_df
final_df = pd.merge(combined_data_df, merged_df, left_on='review', right_on='Review', how='left')

# Drop unnecessary columns
final_df.drop(['Review'], axis=1, inplace=True)

# Merge final_df with topic_issue_df to get the "key" column
final_df = pd.merge(final_df, topic_issue_df[['Topic', 'key']], left_on='Topic_Name', right_on='Topic', how='left')

# Rename columns
final_df.rename(columns={
    'date': 'Date',
    'rating': 'Rating',
    'extracted_devresp': 'Extracted_Dev_Response',
    'devresp_time': 'Dev_Response_Time',
    'clean_review': 'Clean_Review',
    'Positive Score': 'Positive_Score',
    'Negative Score': 'Negative_Score',
    'Neutral Score': 'Neutral_Score',
    'Compound Score': 'Compound_Score',
    'key': 'Issue',
    'nps_indiv': 'NPS_Score_Indiv',
    'nps_category': 'NPS_Category',
}, inplace=True)

# Reorder columns
final_df = final_df[['dataFrom', 'Date', 'review_x', 'Rating', 'Extracted_Dev_Response', 'Dev_Response_Time', 'Clean_Review', 'Topic_Number', 'Topic_Name', 'Issue', 'Positive_Score', 'Negative_Score', 'Neutral_Score', 'Compound_Score', 'NPS_Score_Indiv', 'NPS_Category']]

# Drop duplicates based on the review column
final_df.drop_duplicates(subset='review_x', inplace=True)

# Save the final DataFrame to a new CSV file
final_df.to_csv("final_data.csv", index=False)
