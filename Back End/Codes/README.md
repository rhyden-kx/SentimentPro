# Back End

## Run Requirements.txt
```
pip install requirements.txt
```

## Obtaining Raw Customer Reviews
To obtain customer reviews from both App Store & Google Play Store as csv files respectively. 

Run Codes/AppStoreScrape/AppStoreScrape.py to obtain Back End/Data/AppStoreData.csv

Follow the README.md and then Codes/PlayStoreScrape/JsonToCSV.mjs to obtain Back End/Data/PlayStoreData.csv


## Combining & Cleaning of Data
To combine both App Store & Google Play Store customer reviews into a single csv file: 
Run CombiningData.ipynb to obtain Data/combined_data.csv

To remove all non-alphabetical characters such as emojis:
Run Codes/datacleaning.ipynb to obtain Data/Reviews.csv

## Topic Modelling
To assign topics to each review, run Codes/LDA Topic.ipynb

To obtain accuracy comparison between LdaMallet vs LdaModel, run Codes/accuracy_comparison_lda.ipynb

## Sentiment Analysis
To check NPS accuracy of the model, run Codes/accuracytestfornps.ipynb

To obtain NPS for a topic or subtopic, run Codes/topic_subtopic_npsfunctions.ipynb


## Obtaining Issues & Summaries 
To obtain the summaries/improvements for specific topic
Run Codes/summary.ipynb to obtain a list of improvements recommended by the Generative AI

To obtain the summaries/improvements for specific subtopics
Run Codes/topic_issue.ipynb to obtain Data/subtopic.csv





