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

# Step 1: Extract Locations using spaCy NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# List of known non-locations to exclude
NON_LOCATIONS = {'n’t', 'reddit', 'er', 'dont', 'cant', 'wont', 'im', 'ive', 'id'}

def extract_locations(text):
    if not isinstance(text, str):
        return None
    
    # First, try a simple regex to catch patterns like "in [Location]"
    regex_match = re.search(r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text, re.IGNORECASE)
    if regex_match:
        location = regex_match.group(1)
        if location.lower() not in NON_LOCATIONS:
            return location

    # Fallback to spaCy NER
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    for loc in locations:
        if loc.lower() not in NON_LOCATIONS:
            return loc
    return None

df['Location'] = df['Content'].apply(extract_locations)

# Step 2: Geocode Locations
geolocator = Nominatim(user_agent="crisis_mapping_app")

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

# Geocode locations with a delay to avoid rate limits
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