import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

#load ai model 
llm = OllamaLLM(model="mistral")

#INTIALIZE CHAT HISTORY
chat_history = ChatMessageHistory()

#initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)

#speech recognition  
recognizer = sr.Recognizer()

#function to speak. 
def speak(text):
    engine.say(text)
    engine.runAndWait()

#function to listen 
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
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])

    #Run the AI response generation 
    response = llm.invoke(prompt_template.format(chat_history=chat_history_text, user_input=user_input))

    #store user input and AI response in chat history
    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(response)

    return response

#main loop  
speak("Hello! I am your AI voice assistant. You can ask me anything. Say 'exit' or 'quit' to end the conversation.")
while True:
    user_input = listen()
    if user_input is None:
        continue
    if user_input.lower() in ["exit", "quit"]:
        speak("Exiting the AI voice assistant. Goodbye!")
        break
    response = run_chain(user_input)
    print(f"AI: {response}")
    speak(response)