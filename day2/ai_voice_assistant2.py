import streamlit as st
import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

#load ai model
llm = OllamaLLM(model="mistral")

#initialize the memory (langchain )
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

#initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 200)

#speech recognition
recognizer = sr.Recognizer()

#function to speak ai responses
def speak(text):
    engine.say(text)
    engine.runAndWait()

#function to listen to user input
def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        user_input = recognizer.recognize_google(audio)
        print(f"You: {user_input}")
        return user_input
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None

#Define AI chat prompt template
prompt_template = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template="Previous conversation:\n{chat_history}\n\nUser: {user_input}\nAI:"
)

#function to process ai response
def run_chain(user_input):
    #retrieve chat history as text manually
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])

    #Run the AI response generation 
    response = llm.invoke(prompt_template.format(chat_history=chat_history_text, user_input=user_input))

    #store user input and AI response in chat history
    st.session_state.chat_history.add_user_message(user_input)
    st.session_state.chat_history.add_ai_message(response)

    return response

# Streamlit UI
st.title("AI Voice Assistant with Memory")
st.write("Ask me anything! You can type or use voice input.")

#button to trigger voice input
if st.button("Speak"):
    user_input = listen()
    if user_input:
        response = run_chain(user_input)
        st.write(f"AI: {response}")
        speak(response)

#display full chat history
st.subheader("Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"{msg.type.capitalize()}: {msg.content}")