###Basic AI agent with web ui
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

#load ai model
llm = OllamaLLM(model="mistral")

#Intialize Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

#Define AI chat prompt template
prompt_template = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template="Previous conversation:\n{chat_history}\n\nUser: {user_input}\nAI:"
)

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
st.title("AI Chatbot with Memory")
st.write("Ask me anything!")

user_input = st.text_input("You:", key="user_input")
if user_input:
    response = run_chain(user_input)
    st.write(f"AI: {response}")

#show full chat history
st.subheader("Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"{msg.type.capitalize()}: {msg.content}")

    