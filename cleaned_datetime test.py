import os
import pandas as pd

os.chdir("C:/Users/rhyde/SentimentPro/Data")

import pandas as pd

# Read the main CSV file
main_df = pd.read_csv("merged_score_df_3.csv")

# Read the second CSV file
topic_df = pd.read_csv("topic issue combined.csv")

# Merge the two DataFrames on the 'review' column
merged_df = pd.merge(main_df, topic_df, on='review', how='inner')

# Write the merged DataFrame to a new CSV file
merged_df.to_csv("merged_score_df_final_inner.csv", index=False)