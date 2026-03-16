pip install pyttsx3

# Text to Speech using pyttsx3

import pyttsx3

# Initialize the speech engine
engine = pyttsx3.init()

# Set properties
engine.setProperty('rate', 170)      # Speed of speech
engine.setProperty('volume', 0.9)    # Volume level (0.0 to 1.0)

# Change voice
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)   # Use voices[0] or voices[1]

# Take input from user
text = input("Enter the text to convert to speech: ")

# Speak the text
engine.say(text)
engine.runAndWait()
