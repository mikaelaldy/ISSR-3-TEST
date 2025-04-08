import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Download required NLTK data
nltk.download('punkt')

# Load the cleaned dataset
INPUT_PATH = "../data/reddit_mental_health_cleaned.csv"
OUTPUT_PATH = "../data/reddit_mental_health_analyzed.csv"
PLOT_OUTPUT_DIR = "../visualizations/"

df = pd.read_csv(INPUT_PATH)

# Ensure 'Cleaned_Content' is a string and handle missing values
df['Cleaned_Content'] = df['Cleaned_Content'].astype(str).fillna('')

# Step 1: Sentiment Analysis with VADER
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Cleaned_Content'].apply(get_sentiment)

# Step 2: Risk Classification using TF-IDF and Keyword Matching
# Define high-risk and moderate-risk keywords
HIGH_RISK_KEYWORDS = [
    "want to die", "end my life", "suicidal thoughts", "dont want to be here",
    "kill myself", "cant go on", "no reason to live"
]
MODERATE_RISK_KEYWORDS = [
    "feeling overwhelmed", "need support", "cope with stress", "feeling alone",
    "mental health struggle", "relapse", "depression help", "addiction help"
]

# Use TF-IDF to identify important terms in each post
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Cleaned_Content'])
feature_names = vectorizer.get_feature_names_out()

# Function to classify risk level
def classify_risk(text):
    text_lower = text.lower()
    # Check for high-risk phrases
    for phrase in HIGH_RISK_KEYWORDS:
        if phrase in text_lower:
            return 'High-Risk'
    # Check for moderate-risk phrases
    for phrase in MODERATE_RISK_KEYWORDS:
        if phrase in text_lower:
            return 'Moderate Concern'
    # Default to low concern if no crisis language is detected
    return 'Low Concern'

df['Risk_Level'] = df['Content'].apply(classify_risk)  # Use raw 'Content' to catch phrases before preprocessing

# Step 3: Generate Distribution Table and Plots
# Distribution table
sentiment_risk_dist = pd.crosstab(df['Sentiment'], df['Risk_Level'])
print("\nDistribution of Posts by Sentiment and Risk Level:")
print(sentiment_risk_dist)

def save_and_show_plot(plt, filename):
    """Helper function to save and display plots"""
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Clean up memory

# Plot 1: Sentiment Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Sentiment', palette='viridis')
plt.title('Distribution of Posts by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Number of Posts')
save_and_show_plot(plt, 'sentiment_distribution.png')

# Plot 2: Risk Level Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Risk_Level', palette='magma')
plt.title('Distribution of Posts by Risk Level')
plt.xlabel('Risk Level')
plt.ylabel('Number of Posts')
save_and_show_plot(plt, 'risk_level_distribution.png')

# Plot 3: Sentiment vs Risk Level (Heatmap)
plt.figure(figsize=(8, 5))
sns.heatmap(sentiment_risk_dist, annot=True, fmt='d', cmap='Blues')
plt.title('Sentiment vs Risk Level Distribution')
save_and_show_plot(plt, 'sentiment_vs_risk_heatmap.png')

# Save the updated dataset with sentiment and risk levels
df.to_csv(OUTPUT_PATH, index=False)
print("Updated dataset saved to 'data/reddit_mental_health_analyzed.csv'.")