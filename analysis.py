import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import ssl
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.point import Point
import certifi
import folium
from folium.plugins import HeatMap
import time

# Handle SSL context issue
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK resources
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.update({'like', 'just', 'really', 'actually', 'also', 'would', 'could', 'should', 'best', 'improve', 'students', 'great', 'good', 'better', 'one', 'many', 'group', 'maybe','kid','kids', 'groups', 'loved','enjoyed','experience','make','little', 'part','everything','nothing','fun'})

# Read File
df = pd.read_excel("_Field Trip Survey Data 24-25.xlsx")



# Step 1: Replace "PK" with "-1" and "K" with "0"
df['Cleaned_Grade'] = df['Grade'].str.upper().replace(r'\bPK\b', '-1', regex=True)
df['Cleaned_Grade'] = df['Cleaned_Grade'].str.lower().replace(r'\bK\b', '0', regex=True)

# Step 2: Handle grade ranges (e.g., "4 - 12")
# Expand ranges like "4 - 12" into lists of integers [4, 5, 6, ..., 12]
def expand_ranges(grade_str):
    if pd.isna(grade_str):
        return []
    grades = []
    # Split by commas to handle multiple entries
    for part in grade_str.split(','):
        # Check for range using "start - end" pattern
        match = re.match(r'(-?\d+)\s*-\s*(-?\d+)', part.strip())
        if match:
            start, end = map(int, match.groups())
            grades.extend(range(start, end + 1))
        elif re.match(r'-?\d+', part.strip()):
            grades.append(int(part.strip()))
    return grades

df['Cleaned_Grade'] = df['Cleaned_Grade'].apply(expand_ranges)

# Step 3: Calculate the average grade for each school (row)
df['Row_Average'] = df['Cleaned_Grade'].apply(
    lambda x: np.mean(x) if len(x) > 0 else np.nan
)

df.to_csv('[Cleaned] Field Trip Survey Data 24-25.csv', index=False)
print("Data cleaned and saved to [Cleaned] Field Trip Survey Data 24-25.csv")

cleaned_df = pd.read_csv('[Cleaned] Field Trip Survey Data 24-25.csv')


# Step 4: Calculate the overall average of the row-level averages
overall_average = df['Row_Average'].mean()
print("Average Grade: ", overall_average)

# Step #5 Sentiment analysis
sid = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    if pd.isna(text):
        return None
    sentiment = sid.polarity_scores(text)
    return round(sentiment['compound'], 1)

def get_textblob_sentiment(text):
    if pd.isna(text):
        return None
    analysis = TextBlob(text)
    return round(analysis.sentiment.polarity, 1)

def apply_sentiment_analysis(df, column):
    
    df[f'{column}_VADER_Sentiment'] = df[column].apply(lambda x: get_vader_sentiment(str(x)))
    df[f'{column}_TextBlob_Sentiment'] = df[column].apply(lambda x: get_textblob_sentiment(str(x)))

    # df['VADER_Sentiment_Description'] = df['Combined_VADER_Score'].apply(get_sentiment_description)
    # df['TextBlob_Sentiment_Description'] = df['Combined_TextBlob_Score'].apply(get_sentiment_description)

    return df

# def get_sentiment_description(score):
#     if score is None:
#         return 'No sentiment'
#     elif score <= -0.1:
#         return 'Negative'
#     elif score <= 0.1:
#         return 'Neutral'
#     else:
#         return 'Positive'

df_analyzed = apply_sentiment_analysis(cleaned_df, 'What was the best part of the Hiller Aviation Museum HANDS-ON program?')
df_analyzed = apply_sentiment_analysis(cleaned_df, 'How could the HANDS-ON program be improved?')
df_analyzed = apply_sentiment_analysis(cleaned_df, 'What was the best part of the Hiller Aviation Museum TOUR experience?')
df_analyzed = apply_sentiment_analysis(cleaned_df, 'How could the TOUR experience be improved?')
df_analyzed = apply_sentiment_analysis(cleaned_df, 'What was the best part of the Hiller Aviation Museum experience?')
df_analyzed = apply_sentiment_analysis(cleaned_df, 'How could the experience be improved?')


# Calculate the combined sentiment scores
df_analyzed['Combined_VADER_Sentiment'] = df_analyzed[[col for col in df_analyzed.columns if 'VADER_Sentiment' in col]].mean(axis=1).round(3)
df_analyzed['Combined_TextBlob_Sentiment'] = df_analyzed[[col for col in df_analyzed.columns if 'TextBlob_Sentiment' in col]].mean(axis=1).round(3)

# Calculate the average of the combined sentiment scores
df_analyzed['Average_Combined_Sentiment'] = df_analyzed[['Combined_VADER_Sentiment', 'Combined_TextBlob_Sentiment']].mean(axis=1).round(3)
print("Average Sentiment for all feedback: ", df_analyzed['Average_Combined_Sentiment'].mean().round(3))

df_analyzed.drop(columns = [col for col in df_analyzed.columns if ('VADER_Sentiment' in col or 'TextBlob_Sentiment' in col) and 'Combined' not in col], inplace = True)
df_analyzed.to_csv('[Analyzed] Field Trip Survey Data 24-25.csv', index=False)
print("Data analyzed and saved to [Analyzed] Field Trip Survey Data 24-25.csv")

# Create a custom SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
# Initialize geolocator with Nominatim
geolocator = Nominatim(user_agent="school_geocoder")

# Generate word clouds
def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.savefig(f'{title}.png')
    plt.show()

# Combine text for columns with "improved" in their names
improved_text = ' '.join(df_analyzed[col].dropna().astype(str).str.cat(sep=' ') for col in df_analyzed.columns if 'improved' in col.lower())

# Combine text for columns with "best part" in their names
best_part_text = ' '.join(df_analyzed[col].dropna().astype(str).str.cat(sep=' ') for col in df_analyzed.columns if 'best part' in col.lower())

# Generate and display word clouds
generate_word_cloud(improved_text, 'World Cloud of Areas for Improvement')
print("Areas for Improvement has been saved to Areas for Improvement.png")
generate_word_cloud(best_part_text, 'Word Cloud for Most Enjoyed Parts')
print("Most Enjoyed Parts has been saved to World Cloud for Most Enjoyed Parts.png")



# Create a custom SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Initialize geolocator with Nominatim and custom SSL context
geolocator = Nominatim(user_agent="school_geocoder", ssl_context=ssl_context, timeout=10)

# Geocoding function to get latitude and longitude
def get_geocode(address, geolocator, retries=3):
    try:
        location = geolocator.geocode(address, viewbox = [Point(37.19, -122.60), Point(38.03, -121.73)], bounded = True)
        if location:
            return location.latitude, location.longitude
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        print(f"Error geocoding {address}: {e}")
        if retries > 0:
            time.sleep(1)
            return get_geocode(address, geolocator, retries - 1)
    return None, None

# Get geographical coordinates for each school
df_analyzed['Latitude'], df_analyzed['Longitude'] = zip(*df_analyzed['School'].apply(lambda x: get_geocode(x, geolocator)))

# Drop rows with missing coordinates
df_analyzed.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Create a geographical heatmap of the Bay Area using folium
bay_area_map = folium.Map(location=[37.7749, -122.4194], zoom_start=10)  # Centered around San Francisco

# Add heatmap layer
heat_data = [[row['Latitude'], row['Longitude']] for index, row in df_analyzed.iterrows()]
HeatMap(heat_data).add_to(bay_area_map)

# Save the map to an HTML file
bay_area_map.save('Hiller Schools Heatmap.html')

print("Heatmap saved to Hiller Schools Heatmap.html")
print(df_analyzed['School'].value_counts())