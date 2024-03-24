import streamlit as st
import speech_recognition as sr
import requests
from bs4 import BeautifulSoup
import spacy
from collections import Counter
from heapq import nlargest
import string
import datetime
from deep_translator import GoogleTranslator
from pyAudioAnalysis import audioSegmentation

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

def recognize_speech_from_file(audio_file_path, language='en-IN'):
    """
    Recognizes speech input from an audio file and performs speaker diarization.

    Args:
        audio_file_path (str): Path to the audio file.
        language (str, optional): Language code for speech recognition. Defaults to 'en-IN' (English India).

    Returns:
        str: Recognized text with speaker labels, or None if unable to recognize.
    """
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)  # Read the entire audio file

    # Perform speaker diarization using pyAudioAnalysis
    segments = audioSegmentation.speaker_diarization(audio_file_path)

    recognized_text = []
    previous_speaker = None

    for seg in segments:
        speaker = seg[2]
        speech = seg[3]

        if speaker != previous_speaker:
            recognized_text.append(f"\nSpeaker {speaker}:")  # Add speaker label
            previous_speaker = speaker

        recognized_text.append(speech)

    return " ".join(recognized_text)

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

def search_web(translated_text):
    """
    Searches the web for news articles related to the translated text.

    Args:
        translated_text (str): Translated text to use as the search topic.

    Returns:
        list: List of news article snippets.
    """
    url = f"https://www.google.com/search?q={translated_text}&tbm=nws"
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

def save_minutes(recognized_text, filename="meeting_minutes.txt"):
    """
    Saves meeting minutes, including speaker labels, to a text file.

    Args:
        recognized_text (str): Recognized text with speaker labels.
        filename (str, optional): Name of the output file. Defaults to "meeting_minutes.txt".
    """

    timestamp = get_timestamp()
    with open(f"{filename}-{timestamp}", "w") as file:
        file.write(recognized_text)

def translate_text(text, dest_language='en'):
    """
    Translates text to the target language using Google Translator.

    Args:
        text (str): Text to be translated.
        dest_language (str, optional): Target language code. Defaults to 'en' (English).

    Returns:
        str: Translated text.
    """
    translated_text = GoogleTranslator(source='hi-EN', target=dest_language).translate(text)
    return translated_text

# Streamlit app
st.title("Speech to Text Translation and Analysis")

# Audio file upload
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav', start_time=0)

    recognized_text = recognize_speech_from_file(uploaded_file)
    if recognized_text:
        st.write("Recognized Text with Speaker Labels:")
        st.write(recognized_text)

        translated_text = translate_text(recognized_text, dest_language='en')
        if translated_text:
            st.write("Translated Text:", translated_text)

            preprocessed_text = preprocess_text(translated_text)
            st.write("Preprocessed Text:", preprocessed_text)

            keywords = extract_keywords(preprocessed_text)
            st.write("Keywords:", keywords)

            summary = generate_summary(translated_text)
            st.write("Summary:", summary)

            news_articles = search_web(translated_text)
            st.write("News Articles:", news_articles)

            # Saving meeting minutes to a file
            save_minutes(recognized_text)
            st.write("Meeting minutes with speaker labels saved to file.")
