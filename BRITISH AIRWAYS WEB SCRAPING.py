# Import required libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


# Web Scraping Function
def scrape_skytrax_reviews():
    base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
    pages = 10  # You can increase this number to get more reviews
    page_size = 100
    reviews = []

    for i in range(1, pages + 1):
        print(f"Scraping page {i}")
        url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

        try:
            time.sleep(2)  # Be respectful to the server
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract reviews
            review_containers = soup.find_all("div", {"class": "text_content"})

            # Extract review text
            for container in review_containers:
                review_text = container.get_text().strip()
                reviews.append(review_text)

            print(f"Collected {len(reviews)} reviews so far...")

        except Exception as e:
            print(f"Error on page {i}: {e}")
            continue

    return reviews


# Data Cleaning Function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


# Sentiment Analysis Function
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


# Topic Modeling Function
def perform_topic_modeling(documents, num_topics=5, num_words=10):
    # Create document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(documents)

    # Create and fit LDA model
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_output = lda_model.fit_transform(doc_term_matrix)

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Get top words for each topic
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(top_words)

    return topics


# Main execution
if __name__ == "__main__":
    # 1. Scrape Reviews
    print("Starting web scraping...")
    reviews = scrape_skytrax_reviews()

    # 2. Create DataFrame
    df = pd.DataFrame({'review_text': reviews})

    # 3. Clean the text
    print("Cleaning text...")
    df['cleaned_text'] = df['review_text'].apply(clean_text)

    # 4. Perform Sentiment Analysis
    print("Performing sentiment analysis...")
    df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment)

    # 5. Generate WordCloud
    print("Generating word cloud...")
    all_text = ' '.join(df['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Reviews')
    plt.savefig('wordcloud.png')

    # 6. Perform Topic Modeling
    print("Performing topic modeling...")
    topics = perform_topic_modeling(df['cleaned_text'].tolist())

    # 7. Save results
    # Save DataFrame
    df.to_csv('british_airways_reviews.csv', index=False)

    # Save topic modeling results
    with open('topics.txt', 'w') as f:
        for idx, topic in enumerate(topics):
            f.write(f"Topic {idx + 1}: {', '.join(topic)}\n")

    # 8. Generate some basic statistics
    print("\nBasic Statistics:")
    print(f"Total number of reviews collected: {len(df)}")
    print(f"Average sentiment score: {df['sentiment'].mean():.2f}")
    print(f"Positive reviews: {len(df[df['sentiment'] > 0])} ({(len(df[df['sentiment'] > 0]) / len(df) * 100):.1f}%)")
    print(f"Negative reviews: {len(df[df['sentiment'] < 0])} ({(len(df[df['sentiment'] < 0]) / len(df) * 100):.1f}%)")

    # 9. Create sentiment distribution plot
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='sentiment', bins=50)
    plt.title('Distribution of Review Sentiments')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Count')
    plt.savefig('sentiment_distribution.png')

    print("\nAnalysis complete! Check the generated files for results.")