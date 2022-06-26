import pandas as pd
import requests
import matplotlib.pyplot as plt
import bs4 as BeautifulSoup
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.set_option('deprecation.showPyplotGlobalUse', False)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')


# get the words from a wikipedia artcile from the topic and remove the stopwords and remove the string punctuation
def get_words(topic):
    url = 'https://en.wikipedia.org/wiki/' + topic
    response = requests.get(url)
    soup = BeautifulSoup.BeautifulSoup(response.text, 'html.parser')
    text = soup.find('div', {'id': 'mw-content-text'}).text
    tokens = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    filtered_words = [w for w in tokens if w not in stopwords]
    filtered_words = [w for w in filtered_words if w.isalpha()]
    return filtered_words


# create a dataframe with the words and their frequencies and return the top 50 words
def create_df(topic):
    words = get_words(topic)
    freq = nltk.FreqDist(words)
    df = pd.DataFrame(freq.most_common(), columns=['word', 'freq'])
    return df.head(50)

# create a plot with the words and their frequencies from the dataframe and display in streamlit
def create_plot(df):
    df.plot(x='word', y='freq', kind='bar')
    st.pyplot()
    
# take the words from the dataframe as a list and with nltk use the vader sentiment analyzer to determine the sentiment of the dataframe
def sentiment(df):
    words = df['word'].tolist()
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for word in words:
        sentiments.append(analyzer.polarity_scores(word))
    return sentiments

# take the sentiment of the words and display the sentiment of the topic as a pie chart in streamlit
def display_sentiment(sentiment):
    positive = 0
    negative = 0
    neutral = 0
    for s in sentiment:
        if s['compound'] > 0.05:
            positive += 1
        elif s['compound'] < -0.05:
            negative += 1
        else:
            neutral += 1
    labels = 'Positive', 'Negative', 'Neutral'
    sizes = [positive, negative, neutral]
    colors = ['green', 'red', 'yellow']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')
    st.pyplot()


# create a streamlit app that gets words from wikipedia and displays the words, frequencies, plot, and sentiment of the topic as a pie chart
def create_app():
    st.title('Wikipedia Word Frequency')
    topic = st.text_input('Enter a topic:')
    df = create_df(topic)
    create_plot(df)
    display_sentiment(sentiment(df))
    st.write(df)


# Non-Github Copilot Code
st.title("Github Copilot Test")
st.write("The Github repo can be found here: https://github.com/PeterMorrison1/Copilot_Wikipedia_Sentiment_Analyzer")
st.write("This project is to demonstrate the use of Github Copilot. All comments are written by myself and all code is written by Github Copilot.")
st.write("I couldn't get Copilot to lowercase the words as they were read from Wikipedia. I think a better practice would be to make very atomic functions as Copilot can't handle the complexity of the request.")
st.write("I also had to specify streamlit charts, otherwise it would give matplotlib charts.")
st.write("I also had to import libraries manually.")
# End of Non-Github Copilot Code


create_app()