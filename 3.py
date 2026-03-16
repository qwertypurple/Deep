pip install SpeechRecognition pyaudio


# Program 2: Speech Recognition using SpeechRecognition

import speech_recognition as sr

# Audio file name
AUDIO_FILE = "audio.wav"

# Create recognizer object
recognizer = sr.Recognizer()

# Load audio file
with sr.AudioFile(AUDIO_FILE) as source:
    audio = recognizer.listen(source)

# Convert speech to text using Google API
text = recognizer.recognize_google(audio)

# Print recognized text
print("Recognized Speech:")
print(text)
