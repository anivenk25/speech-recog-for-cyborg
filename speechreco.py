import speech_recognition as sr
import requests
from bs4 import BeautifulSoup
import spacy
from collections import Counter
from heapq import nlargest
import string
import datetime

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")


def recognize_speech(language='en-IN'):
  """
  Recognizes speech input from the microphone.

  Args:
      language (str, optional): Language code for speech recognition. Defaults to 'en-IN' (English India).

  Returns:
      str: Recognized text, or None if unable to recognize.
  """

  recognizer = sr.Recognizer()

  with sr.Microphone() as source:
    print(f"Listening for {language} speech...")
    recognizer.adjust_for_ambient_noise(source, duration=1)
    audio = recognizer.listen(source)

  try:
    if language == 'en-IN':
      recognized_text = recognizer.recognize_google(audio, language=language)
    elif language == 'hi-IN':
      recognized_text = recognizer.recognize_google(audio, language=language)
    elif language == 'or-IN':
      print("Listening for Odia speech...")
    else:
      recognized_text = None
      print("Language not supported")
  except sr.UnknownValueError:
    recognized_text = None
    print("Unable to recognize speech")

  return recognized_text

def preprocess_text(text):
  """
  Preprocesses text by tokenizing, lemmatizing, and removing stop words and punctuation.

  Args:
      text (str): Text to be preprocessed.

  Returns:
      list: List of preprocessed tokens.
  """

  doc = nlp(text)
  tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
  return tokens

def extract_keywords(text, num_keywords=5):
  """
  Extracts top keywords from text using word frequency.

  Args:
      text (str): Text to extract keywords from.
      num_keywords (int, optional): Number of keywords to extract. Defaults to 5.

  Returns:
      list: List of top keywords.
  """

  word_freq = Counter(text)
  return nlargest(num_keywords, word_freq, key=word_freq.get)

def generate_summary(text, num_sentences=3):
  """
  Generates a summary of the text using spaCy's built-in summarization feature.

  Args:
      text (str): Text to summarize.
      num_sentences (int, optional): Number of sentences to include in the summary. Defaults to 3.

  Returns:
      str: Summary of the text.
  """

  doc = nlp(text)
  sentences = [sent.text for sent in doc.sents]
  summary = " ".join(sentences[:num_sentences])
  return summary

def search_web(topic):
  """
  Searches the web for news articles related to the provided topic.

  Args:
      topic (str): Topic to search for.

  Returns:
      list: List of news article snippets.
  """

  url = f"https://www.google.com/search?q={topic}&tbm=nws"
  headers = {"User-Agent": "Mozilla/5.0"}
  response = requests.get(url, headers=headers)
  soup = BeautifulSoup(response.text, 'html.parser')
  news_results = soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd')
  return [result.get_text() for result in news_results]

def get_timestamp():
  """
  Returns a timestamp in YYYY-MM-DD format.

  Returns:
      str: Timestamp.
  """

  return datetime.datetime.now().strftime('%Y-%m-%d')

def save_minutes(meeting_minutes, filename="meeting_minutes.txt"):
  """
  Saves meeting minutes to a text file, with timestamps for each speech segment.

  Args:
      meeting_minutes (list): List of meeting speech segments.
      filename (str, optional): Name of the output file. Defaults to "meeting_minutes.txt".
  """

  timestamp = get_timestamp()
  with open(f"{filename}-{timestamp}", "w") as file:
    for i, speech in enumerate(meeting_minutes):
      file.write
