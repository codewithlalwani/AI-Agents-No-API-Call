
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


#load ai model
llm = OllamaLLM(model="mistral")


#INTIALIZE CHAT HISTORY
chat_history = ChatMessageHistory()

#Define AI chat prompt template
prompt_template = PromptTemplate( 
    input_variables=["chat_history", "user_input"],
    template="Previous conversation:\n{chat_history}\n\nUser: {user_input}\nAI:"
)

#function to run ai chat with memory 
def run_chain(user_input):
    #retrieve chat history as text manually
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])

    #Run the AI response generation 
    response = llm.invoke(prompt_template.format(chat_history=chat_history_text, user_input=user_input))

    #store user input and AI response in chat history
    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(response)

    return response

#inteactive cli chatbot 
print("\n AI Chatbot with memory")
print("Type 'exit' or 'quit' to end the conversation.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the AI Chatbot. Goodbye!")
        break
    response = run_chain(user_input)
    print(f"AI: {response}")





# from langchain_ollama import OllamaLLM
# llm = OllamaLLM(model="mistral")

# print("Welcome to the Basic AI Agent!, Ask me anything and I will try to answer your questions.") 
# while True:
#     user_input = input("\nYou: ")
#     if user_input.lower() in ["exit", "quit"]:
#         print("Exiting the Basic AI Agent. Goodbye!")
#         break
#     response = llm.invoke(user_input)
#     print(f"AI: {response}")