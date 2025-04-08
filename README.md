
# ISSR-3-TEST

This project extracts, analyzes, and maps social media posts related to mental health distress, substance use, or suicidality. It consists of three main tasks:

1. **Task 1**: Extract and preprocess social media data (Reddit posts) related to mental health crises.
2. **Task 2**: Perform sentiment analysis and crisis risk classification on the extracted posts.
3. **Task 3**: Geocode and map the locations of posts to identify crisis hotspots.

The project was initially built using Reddit data (via the Reddit API with `praw`), but due to challenges with geolocation, there are plans to redo the project using Twitter data with the Tweet-Harvest CLI tool.

## Project Structure

- `scripts/`: Contains the Python scripts for each task.
  - `task1_data_extraction.py`: Extracts and preprocesses Reddit posts.
  - `task2_sentiment_risk.py`: Analyzes sentiment and classifies risk levels.
  - `task3_geolocation_mapping.py`: Geocodes locations and generates a heatmap.
- `data/`: Stores the raw and processed datasets.
  - `reddit_mental_health_cleaned.csv`: Cleaned data from Task 1.
  - `reddit_mental_health_analyzed.csv`: Data with sentiment and risk levels from Task 2.
  - `reddit_mental_health_geocoded.csv`: Data with geolocation from Task 3.
- `visualizations/`: Stores the output plots and heatmap.
  - `sentiment_distribution.png`: Distribution of posts by sentiment.
  - `risk_level_distribution.png`: Distribution of posts by risk level.
  - `sentiment_vs_risk_heatmap.png`: Heatmap of sentiment vs. risk level.
  - `crisis_heatmap.html`: Interactive heatmap of crisis locations.

## Setup Instructions

Follow these steps to clone the repository and set up the project on your local machine.

### Prerequisites

- Python 3.8 or higher
- Git
- A Reddit API account (for Task 1, if using Reddit data)
- Internet access (for API calls and geocoding)

### Step 1: Clone the Repository

Clone the repository to your local machine using Git:

```bash
git clone git@github.com:mikaelaldy/ISSR-3-TEST.git
cd ISSR-3-TEST
```

### Step 2: Install Dependencies

Install the required Python packages listed in requirements.txt:

```shell
pip install -r requirements.txt
```

### Step 3: Install spaCy Model

Install the spaCy English model used for named entity recognition (NER) in Task 3:

```bash
python -m spacy download en_core_web_sm
```

Then update task3_geolocation_mapping.py to use nlp = spacy.load("en_core_web_md").

### Step 4: Configure Reddit API Credentials

- Create a Reddit app at [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps).
- Note your client_id, client_secret, and user_agent.
- Create a config.py file in the scripts/ directory with the following content:

```bash
REDDIT_CLIENT_ID = "your_client_id"
REDDIT_CLIENT_SECRET = "your_client_secret"
REDDIT_USER_AGENT = "your_user_agent"
```

## Running the Scripts

Run each script in sequence to replicate the work.

### Task 1: Data Extraction and Preprocessing

Extracts Reddit posts related to mental health distress, substance use, or suicidality, and preprocesses the text.

```bash
python scripts/task1_data_extraction.py
```

**Output**:

- ../data/reddit_mental_health_cleaned.csv: Cleaned dataset with columns like PostID, Timestamp, Subreddit, Score, Comments, URL, Cleaned_Content, Title, and Content.

### Task 2: Sentiment and Crisis Risk Classification

Analyzes the sentiment of posts using VADER and classifies them into risk levels (High-Risk, Moderate Concern, Low Concern).

```bash
python scripts/task2_sentiment_risk.py
```

**Output**:

- ../data/reddit_mental_health_analyzed.csv: Updated dataset with Sentiment and Risk_Level columns.
- Visualizations in ../visualizations/:
  - sentiment_distribution.png: Bar chart of sentiment distribution.
  - risk_level_distribution.png: Bar chart of risk level distribution.
  - sentiment_vs_risk_heatmap.png: Heatmap of sentiment vs. risk level.

### Task 3: Geolocation and Mapping

Extracts locations from post content, geocodes them, and generates a heatmap of crisis hotspots.

```bash
python scripts/task3_geolocation_mapping.py
```

**Output**:

- ../data/reddit_mental_health_geocoded.csv: Updated dataset with Location, Latitude, and Longitude columns.
- ../visualizations/crisis_heatmap.html: Interactive heatmap showing crisis locations.

## Challenges Faced

### Geolocation with Reddit Data

One of the main challenges was extracting geolocation data from Reddit posts:

- **Problem**: Reddit’s API (praw) does not provide direct access to user location data, and most users do not include their location in their profiles or flairs.
- **Workaround**: I initially used spaCy’s NER to extract locations from post content, but it misclassified non-geographic entities (e.g., "my life," "bed") as locations. I then implemented a regex pattern (\b(?:in|from|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b) to catch phrases like "in London," but this also produced false positives.
- **Solution**: I created a comprehensive list of non-locations (e.g., "my life," "bed") to filter out false positives and added validation using geopy to ensure only geocodable locations were kept. This improved the accuracy but still relied heavily on the quality of the post content.

## Solutions and Future Improvements

### Alternative Data Source: Twitter with Tweet-Harvest

To address the geolocation challenge:

- **Initial Plan**: I considered using the Twitter API, which provides better geolocation data (e.g., user profile locations or geotagged tweets). However, the Twitter API now requires paid credits, which was not feasible for this project.
- **Solution**: I plan to redo the project using the Tweet-Harvest CLI tool, an open-source alternative for scraping Twitter data without API costs. Tweet-Harvest can extract tweets with geolocation data (if available) and user profile locations, which should improve the accuracy of Task 3.
