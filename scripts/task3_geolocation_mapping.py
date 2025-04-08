import pandas as pd
import spacy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import folium
from folium.plugins import HeatMap
import time
import re

# Load the dataset with sentiment and risk levels
INPUT_PATH = "../data/reddit_mental_health_analyzed.csv"
OUTPUT_PATH = "../data/reddit_mental_health_geocoded.csv"
HEATMAP_PATH = "../visualizations/crisis_heatmap.html"

df = pd.read_csv(INPUT_PATH)

# Step 1: Extract Locations using spaCy NER and Regex
# Load spaCy's English model (consider using 'en_core_web_md' for better accuracy)
try:
    nlp = spacy.load("en_core_web_sm")  # Optionally: "en_core_web_md"
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# List of known non-locations to exclude
NON_LOCATIONS = {
    'my life', 'life', 'bed', 'school', 'my', 'the past', 'the moment', 'reality', 'silence',
    'the hospital', 'the mirror', 'the family', 'class', 'the psych ward', 'the back of ambulances',
    'my room', 'the past year', 'the middle of', 'the world', 'the morning', 'the head', 'the public',
    'my consciousness', 'the relationship', 'high school', 'the elementary school lobby', 'the store',
    'my hometown', 'the country', 'the group', 'the wrong place', 'this sub', 'summer', 'the roster',
    'the apartment', 'this world', 'my class', 'general', 'different states', 'this big city',
    'every moment', 'my own life', 'the corners of the room', 'pain minute by minute', 'my own body',
    'this empty routine', 'all of that', 'these few years', 'college in', 'january this year',
    'this life cost money', 'my automatically generated screen name is astounding and wonderfully',
    'the same place', 'my classes', 'my eyes being fat is worse than being depressed', 'mental pain',
    'this house is like and they', 'two weeks with so much left to do and party in the next month otherwise',
    'my sleep', 'any way', 'drunk fits of rage', 'hell', 'my mid', 'years', 'that moment but still somehow exist afterward',
    'hopes that', 'october', 'the beginning was care about the gender and', 'pinocchio', 'my car for',
    'the dumps anybody wanna talk', 'the back of the head before driving off with', 'like', 'silence with',
    'my experience', 'your face that something is off for couple of months', 'was toxic',
    'relationships who only turn to me when there', 'instructions on how to communicate yet', 'real life',
    'person but', 'december and', 'between', 'it', 'dryer which', 'every thing play football',
    'this vast world', 'her life up to', 'my senior year of high school', 'my dream uni',
    'the grand scheme of things', 'the whole world', 'the world where people would just leave me alone',
    'my relationships and life in general', 'my life even if', 'the darkness and have been through',
    'schools is tortuous', 'terms of my depression lately is just the complete lack of desire to do anything at all',
    'skin suits and the fact that', 'my life and got caught up with bad people', 'times my anxiety spikes is',
    'my lates', 'me', 'order to maintain sanity', 'the thick of their panic', 'and feel comfortable and accepted and wanted by people',
    'with my parents', 'my adult life', 'the long run', 'public or social settings', 'healing anxiety',
    'relationships', 'talking to someone who is also going through the struggles of anxiety', 'customer support',
    'terror with chest pressure', 'advance because', 'place with my heart racing and trying to find anyone to reassure me',
    'my life when', 'my daily life at work', 'the home stretch of my junior year of college at', 'pain bye',
    'all this', 'the middle of the night', 'this state', 'the year', 'heaven or hell',
    'the morning she wrote me that she has some problems', 'small doses or will there be', 'my throat',
    'one week', 'the middle of somewhere', 'my head and my brain screams that', 'with them just under',
    'almost', 'the mid afternoon', 'front of boyfriend', 'fear of my next mistake every day at work',
    'my head that tells me', 'even doing all this', 'advance but', 'university and', 'one place',
    'advance', 'college and almost', 'translation', 'the parking lot', 'nursing school and studying with',
    'the system for saying that', 'conjuction with social phobia diagnosis',
    'an especially depressive episode and im too emotionally and mentally exhausted to even think maybe sad reel will make me cry',
    'the past but nothing that', 'relation to', 'your head and that you',
    'love with weed again and forget about drinking and pills', 'january and things were so great between us',
    'doesn\'t', 'constant fear of it getting to the point of an anxiety attack', 'foster care until',
    'full time school', 'fact', 'my head', 'my attempt at', 'my chest', 'sibo', 'time',
    'waves throughour the day and it', 'the coming times', 'my truck listening to mac miller tunes',
    'the back of mind', 'both my psych classes and sociology classes', 'high school for being effeminate',
    'the right way', 'medical school', 'my exams', 'my life that changed everything for me',
    'therapy and to my friends so much but it doesn', 'life since', 'love', 'out of guilt',
    'too many ways', 'so many ways and', 'my first week', 'some definitions', 'general grieving over me terrifies me',
    'bed without paying for it', 'some random university because', 'on it', 'my early',
    'the same state ever since then we', 'these specific people', 'so many ways', 'touch with reality',
    'the hospital where', 'this place', 'jan and', 'sixth form together too', 'love with',
    'her depressive state', 'my feels', 'the best place', 'an abusive household with',
    'high school makes me feel terrible', 'love with my best friend oh my lord im here again about this cause',
    'person', 'those two and', 'trouble and to never respond to her', 'the moment it seems funny',
    'next month', 'my life and narcissistic in many ways', 'my family because thats the only day in the year where you are the main character',
    'everything', 'the military at', 'it are as well', 'every now and then', 'the first place',
    'my living room after the', 'my own way', 'love with him ive', 'one of the intelligence communities',
    'the mental hospital twice and got rid of the help', 'her shadow at times', 'this context is',
    'being kind', 'something worse', 'the friend zone', 'my mouth once', 'me improved but',
    'the way', 'front of us shouting since', 'love nor relationships', 'my family has it',
    'months and', 'hand living in', 'me feeling like shit', 'my own skills', 'hindsight is',
    'the same situation', 'my house', 'her truck with her arguing over radio stations', 'my chest',
    'circles', 'on my dead body so', 'group settings', 'icu after five months',
    'my life that wants to hang out with me edit', 'god and', 'my stupid daydreams',
    'class during this', 'november', 'the last few weeks', 'constant fear of the pain coming back',
    'life it', 'your bones and tells you this is all there will ever be', 'highschool', 'this shit',
    'the same house with my abusive ex partner', 'the mirror and not hate myself',
    'this knowing life is about to end', 'an irrational way', 'so much pain', 'university',
    'awhile', 'these autistic meltdowns and they start spiraling out of control every single day',
    'my ear as', 'front of me and is all sweet', 'silence for', 'my main developmental years',
    'my last relationship', 'my entire life no matter how badly', 'living anymore if this is all that is going to happen to me',
    'carrying on living my wife just passed away', 'his life saw how truly happy he was for the first time ever',
    'my classes and', 'so much pain', 'sth completely unrelated but my depression has made me unable to get',
    'my chest isn', 'mental pain and because of that', 'and out of the psych ward for over half',
    'me is sexual and even then it', 'school and naturally super smart', 'my head is', 'anything',
    'hospital for', 'thousands of dollars in debt trying to seek help and it didn', 'the process of putting them behind me',
    'detail right now', 'your life', 'school growing up and', 'the mirror and get praise for became distorted looking overnight',
    'my room and cry because', 'bed and binge eating', 'overcast', 'bed all day with no motivation and easy tasks are hard for me',
    'so many aspects of myself and my life even though', 'comparison', 'january and the thought of staying in new york state for the next two years is making me want to jump off of',
    'my room', 'countless cities across the us', 'some hobbies or things it makes me feel disgusted with myself and full of hatred',
    'advance for any tips', 'the worst way possible', 'the middle of the night wake to her calls about problems mostly everyday for the past',
    'males the early', 'my room to get away from her', 'trouble', 'the matter and basically just causing',
    'life and confused as to what', 'my room bedrotting and watching youtube vids for hours bc no one',
    'these twelve years it hasn', 'with him', 'my first', 'and become', 'this', 'bandages bc',
    'my mind like', 'my life has gotten worse', 'therapy trying to work through it',
    'my car before class and more money down the drain', 'public will be envious of me',
    'the last few years we', 'therapy and never getting any better', 'my stomach when reacting with media',
    'me isolating myself', 'mental illness and me not being able to see', 'relationship with other adults and other mental health issues',
    'over', 'the field', 'everyone', 'divorce', 'about', 'my grandma',
    'the middle of the day if anyone has some suggestions as to how', 'at the moment',
    'prison his whole life', 'nonstop rumination', 'kinda', 'the book', 'healthy ways and',
    'advance but', 'the attic for years doesn', 'touch with', 'teaching', 'and day out',
    'hopes that the games', 'communication', 'the world do you balance being chronically depressed and hygiene',
    'time set apart for meals and exercise', 'therapy okay', 'recent years', 'few months',
    'my line of work', 'me for not trying hard enough', 'the psych ward because of my weed tattoo'
}

# Initialize geolocator for validation
geolocator = Nominatim(user_agent="crisis_mapping_app")

def validate_location(location):
    """Check if a location can be geocoded; return True if valid, False otherwise."""
    if not location or location.lower() in NON_LOCATIONS:
        return False
    try:
        geo = geolocator.geocode(location, timeout=10)
        return geo is not None
    except (GeocoderTimedOut, Exception):
        return False

def extract_locations(text):
    if not isinstance(text, str):
        return None
    
    # Step 1: Use regex to catch patterns like "in London" or "from India"
    regex_match = re.search(r'\b(?:in|from|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text, re.IGNORECASE)
    if regex_match:
        location = regex_match.group(1)
        if validate_location(location):
            return location

    # Step 2: Fallback to spaCy NER
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    for loc in locations:
        if validate_location(loc):
            return loc
    return None

df['Location'] = df['Content'].apply(extract_locations)

# Step 2: Geocode Locations
def geocode_location(location):
    if not location:
        return None, None
    try:
        geo = geolocator.geocode(location, timeout=10)
        if geo:
            return geo.latitude, geo.longitude
        return None, None
    except (GeocoderTimedOut, Exception) as e:
        print(f"Geocoding error for {location}: {e}")
        return None, None

# Geocode locations (with delay to avoid rate limits)
df['Coordinates'] = df['Location'].apply(lambda x: geocode_location(x))
df['Latitude'] = df['Coordinates'].apply(lambda x: x[0] if x else None)
df['Longitude'] = df['Coordinates'].apply(lambda x: x[1] if x else None)

# Drop rows without coordinates
df_geo = df.dropna(subset=['Latitude', 'Longitude'])

# Step 3: Display Top 5 Locations
location_counts = df['Location'].value_counts().head(5)
print("\nTop 5 Locations with Highest Crisis Discussions:")
print(location_counts)

# Step 4: Generate Heatmap with Folium
m = folium.Map(location=[0, 0], zoom_start=2)
heat_data = [[row['Latitude'], row['Longitude'], 1] for _, row in df_geo.iterrows()]
HeatMap(heat_data).add_to(m)
m.save(HEATMAP_PATH)
print(f"Heatmap saved to '{HEATMAP_PATH}'. Open in a browser to view.")

# Save the dataset with geolocation data
df.to_csv(OUTPUT_PATH, index=False)
print(f"Geocoded dataset saved to '{OUTPUT_PATH}'.")