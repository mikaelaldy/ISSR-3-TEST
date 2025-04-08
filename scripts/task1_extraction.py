import praw
import pandas as pd
import datetime as dt
import re
import emoji # Using 'emoji' library
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time # To potentially add delays

# --- Configuration ---
try:
    # Attempt to import credentials from config.py
    import config
    CLIENT_ID = config.REDDIT_CLIENT_ID
    CLIENT_SECRET = config.REDDIT_CLIENT_SECRET
    USER_AGENT = config.REDDIT_USER_AGENT
except ImportError:
    # Fallback or error if config.py is not found
    print("Error: config.py not found. Please create it with your Reddit API credentials.")
    # You might want to exit or use environment variables as an alternative here
    exit() # Exit if no credentials


KEYWORDS = [
    "depressed", "depression help", "anxiety attack", "feeling overwhelmed",
    "suicidal thoughts", "want to die", "end my life", "lonely",
    "addiction help", "substance abuse", "recovery support", "relapse",
    "mental health struggle", "feeling alone", "need support", "cope with stress"
]
SEARCH_QUERY = " OR ".join(f'"{k}"' for k in KEYWORDS) # Use OR to combine keywords, quotes for exact phrases

# Choose relevant subreddits (be mindful of subreddit rules & sensitivity)
SUBREDDITS = ["mentalhealth", "depression", "anxiety", "addiction", "offmychest", "SuicideWatch", "lonely"] # Example list
POST_LIMIT_PER_SUBREDDIT = 100 # Adjust as needed, mindful of API limits and processing time
OUTPUT_FILENAME_RAW = "../data/reddit_mental_health_raw.csv"
OUTPUT_FILENAME_CLEANED = "../data/reddit_mental_health_cleaned.csv"

stop_words = set(stopwords.words('english'))



def setup_reddit_api():
    """Initializes and returns a PRAW Reddit instance."""
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
        )
        reddit.read_only = True # Good practice if you're only reading data
        print("Reddit API connection successful (Read-Only Mode).")
        return reddit
    except Exception as e:
        print(f"Error connecting to Reddit API: {e}")
        return None

def fetch_posts(reddit, subreddits, query, limit):
    """Fetches posts from specified subreddits based on a query."""
    all_posts_data = []
    if not reddit:
        return all_posts_data

    print(f"Fetching up to {limit} posts per subreddit for query: '{query}'...")
    for sub_name in subreddits:
        print(f"Searching in r/{sub_name}...")
        try:
            subreddit = reddit.subreddit(sub_name)
            # Using search; alternatives exist (e.g., iterating through new posts)
            # Sort by relevance or new - 'new' might be better for recent distress
            submissions = subreddit.search(query, limit=limit, sort='new')

            count = 0
            for post in submissions:
                # Combine title and body text for full content
                content = post.title + " " + post.selftext
                post_data = {
                    "PostID": post.id,
                    "Timestamp": dt.datetime.utcfromtimestamp(post.created_utc),
                    "Subreddit": sub_name,
                    "Title": post.title,
                    "Content": content,
                    "Score": post.score, # Reddit's upvotes/downvotes score
                    "Comments": post.num_comments,
                    "URL": post.url
                }
                all_posts_data.append(post_data)
                count += 1
            print(f"-> Found {count} posts in r/{sub_name}.")
            # Optional: Add a small delay to be nice to the API
            time.sleep(1)

        except praw.exceptions.PRAWException as e:
            print(f"Error accessing subreddit r/{sub_name}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing r/{sub_name}: {e}")

    print(f"Finished fetching. Total posts collected: {len(all_posts_data)}")
    return all_posts_data

def preprocess_text(text):
    """Cleans text data: lowercase, remove URLs, emojis, special chars, stopwords."""
    if not isinstance(text, str):
        return ""
    
    # Combine multiple re.sub operations
    text = text.lower()
    # Combine URL and social media patterns
    text = re.sub(r'(http\S+|www\S+|https\S+|\@\w+|\#)', '', text, flags=re.MULTILINE)
    text = emoji.demojize(text)
    # Combine special characters and numbers removal
    text = re.sub(r'(:[a-zA-Z_]+:|[^\w\s]|\d+)', '', text)
    
    # Tokenize and remove stopwords in one step
    tokens = [word for word in word_tokenize(text) 
             if word not in stop_words and len(word) > 1]
    
    return " ".join(tokens)


# --- Main Execution ---
if __name__ == "__main__":
    reddit_instance = setup_reddit_api()

    if reddit_instance:
        raw_posts = fetch_posts(reddit_instance, SUBREDDITS, SEARCH_QUERY, POST_LIMIT_PER_SUBREDDIT)

        if raw_posts:
            # Create DataFrame
            df = pd.DataFrame(raw_posts)

            # --- Preprocess Content ---
            print("Preprocessing text content...")
            # Ensure 'Content' is string type before applying preprocessing
            df['Content'] = df['Content'].astype(str)
            df['Cleaned_Content'] = df['Content'].apply(preprocess_text)
            print("Preprocessing complete.")

            # Select and reorder columns for final cleaned output
            cleaned_df = df[['PostID', 'Timestamp', 'Subreddit', 'Score', 'Comments', 'URL', 'Cleaned_Content', 'Title', 'Content']] # Keep raw content for reference if needed

            # --- Store Cleaned Data ---
            print(f"Saving cleaned data to {OUTPUT_FILENAME_CLEANED}...")
            cleaned_df.to_csv(OUTPUT_FILENAME_CLEANED, index=False, encoding='utf-8')
            # Alternatively, save to JSON:
            # cleaned_df.to_json("reddit_mental_health_cleaned.json", orient="records", lines=True, date_format="iso")


            print(f"\nFirst 5 rows of cleaned data:\n{cleaned_df.head().to_string()}")
        else:
            print("No posts were fetched. Exiting.")
    else:
        print("Could not establish Reddit connection. Exiting.")