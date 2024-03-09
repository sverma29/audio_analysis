import os
import traceback
import streamlit as st
import speech_recognition as sr
from transformers import pipeline

st.set_page_config(layout="wide")

st.title("Audio Recording Analysis")
st.write("[ENGRO](https://engro.io/)")

st.sidebar.title("Description")
st.sidebar.write("The tool allows one to upload an audio file and perform sentiment analysis")

st.sidebar.header("Upload Audio")
audio_file = st.sidebar.file_uploader("Browse", type=["wav"])
upload_button = st.sidebar.button("Upload")

def perform_sentiment_analysis(text):
  model_name = "distilbert-base-uncased-finetuned-sst-2-english"
  sentiment_analysis = pipeline("sentiment-analysis", model=model_name)
  results = sentiment_analysis(text)
  sentiment_label = results[0]['label']
  sentiment_score = results[0]['score']
  return sentiment_label, sentiment_score


def transcribe_audio(audio_file):
  r = sr.Recognizer()
  with sr.AudioFile(audio_file) as source:
    audio = r.record(source)
    transcribed_text = r.recognize_google(audio)
  return transcribed_text


def main():
  if audio_file and upload_button:
    try:
      transcribed_text = transcribe_audio(audio_file)
      sentiment_label, sentiment_score = perform_sentiment_analysis(transcribed_text)
    
      st.header("Transcribed Text")
      st.text_area("Transcribed Text", transcribed_text, height=200)
      st.header("Sentiment Analysis")
      negative_icon = "👎"
      neutral_icon = "😐"
      positive_icon = "👍"

      if sentiment_label == "NEGATIVE":
        st.write(f"{negative_icon} Negative (Score: {sentiment_score})", unsafe_allow_html=True)
      else:
        st.empty()

      if sentiment_label == "NEUTRAL":
        st.write(f"{neutral_icon} Neutral (Score: {sentiment_score})", unsafe_allow_html=True)
      else:
        st.empty()

      if sentiment_label == "POSITIVE":
        st.write(f"{positive_icon} Positive (Score: {sentiment_score})", unsafe_allow_html=True)
      else:
        st.empty()

    except Exception as ex:
      st.error("Error occurred during audio transcription and sentiment analysis.")
      st.error(str(ex))
      traceback.print_exc()

if __name__ == "__main__": main()