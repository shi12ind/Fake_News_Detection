import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
from textblob import TextBlob
import pandas as pd

# Load Excel files using read_excel
true_df = pd.read_excel(r'/Users/pravinshinde/Documents/True.xlsx')
fake_df = pd.read_excel(r'/Users/pravinshinde/Documents/Fake.xlsx')

# Print the first few rows of each DataFrame
print("True DataFrame:")
print(true_df.head())

print("\nFake DataFrame:")
print(fake_df.head())

fake_df.shape, true_df.shape
fake_df.head()
true_df.head()
fake_df.columns, true_df.columns
true_df.info(), fake_df.info()
fake_df.isnull().sum(), true_df.isnull().sum()
true_df=true_df.drop(columns=['Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'], errors='ignore')
fake_df = fake_df.dropna(subset=['text'])
true_df = true_df.dropna(subset=['text'])
fake_df.isnull().sum(),true_df.isnull().sum()
true_df.duplicated().sum(), fake_df.duplicated().sum()
#drop duplicate data
true_df.drop_duplicates(inplace=True)
fake_df.drop_duplicates(inplace=True)
true_df.duplicated().sum(), fake_df.duplicated().sum()
fake_df.shape, true_df.shape
fake_df['label']=0
true_df['label']=1
data = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
data.shape
data.nunique()
def clean_text(text):
    if not isinstance(text, str):  # Handle non-string values if any remain
        return ""
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

data['cleaned_text'] = data['text'].apply(clean_text)

print(data['cleaned_text'].head())

def get_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    return sentiment

data['sentiment'] = data['cleaned_text'].apply(get_sentiment)
plt.figure(figsize=(10, 6))
sns.histplot(data['sentiment'], bins=30, kde=True, color='blue')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()

def generate_ngrams(text, n=2):
    words = word_tokenize(text)
    n_grams = list(ngrams(words, n))
    return n_grams

fake_text = ' '.join(data[data['label'] == 0]['cleaned_text'])
real_text = ' '.join(data[data['label'] == 1]['cleaned_text'])

fake_bigrams = generate_ngrams(fake_text, n=2)
real_bigrams = generate_ngrams(real_text, n=2)

fake_bigram_freq = Counter(fake_bigrams)
real_bigram_freq = Counter(real_bigrams)

fake_bigram_df = pd.DataFrame(fake_bigram_freq.most_common(10), columns=['Bigram', 'Frequency'])
real_bigram_df = pd.DataFrame(real_bigram_freq.most_common(10), columns=['Bigram', 'Frequency'])

fake_bigram_df['Bigram'] = fake_bigram_df['Bigram'].astype(str)
real_bigram_df['Bigram'] = real_bigram_df['Bigram'].astype(str)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(data=fake_bigram_df, x='Frequency', y='Bigram', color='red', ax=axes[0])
axes[0].set_title('Top 10 Bigrams in Fake News')

sns.barplot(data=real_bigram_df, x='Frequency', y='Bigram', color='green', ax=axes[1])
axes[1].set_title('Top 10 Bigrams in Real News')

plt.tight_layout()
plt.show()

fake_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(fake_text)
real_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(real_text)

plt.figure(figsize=(10, 6))
plt.imshow(fake_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Fake News')
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(real_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Real News')
plt.show()

label_counts = data['label'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(label_counts, labels=['Real News', 'Fake News'], autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
plt.title('Distribution of Fake vs Real News')
plt.show()

data['text_length'] = data['cleaned_text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='text_length', hue='label', bins=30, kde=True, palette={0: 'red', 1: 'green'})
plt.title('Text Length Distribution')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(data=data[data['label'] == 0], x='text_length', label='Fake News', color='red', fill=True, alpha=0.5)
sns.kdeplot(data=data[data['label'] == 1], x='text_length', label='Real News', color='green', fill=True, alpha=0.5)
plt.title("Density Plot of Article Lengths", fontsize=16)
plt.xlabel("Number of Words", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend()
plt.show()

