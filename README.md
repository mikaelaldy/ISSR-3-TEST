## Overview

This project extracts, analyzes, and maps tweets related to mental health distress, substance use, or suicidality.

## Setup

1. Clone the repository: `git clone git@github.com:mikaelaldy/ISSR-3-TEST.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Install spaCy model: `python -m spacy download en_core_web_sm`

## Running the Scripts

-**Task 1**: `python scripts/task1_data_extraction.py`
-**Task 2**: `python scripts/task2_sentiment_risk.py`
-**Task 3**: `python scripts/task3_geolocation_mapping.py`

## Notes

- Ensure internet access for TweetHarvest and geocoding.
- Adjust `tweet_count` in Task 1 if needed.
