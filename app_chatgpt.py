import streamlit as st
import os
#from openai import OpenAI
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Variables
TOKENS_MAXIMOS = 2500


load_dotenv()

# List of models
# OpenAI model
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# llm = OpenAI(api_key=OPENAI_API_KEY)
# modelo = "gpt-4o-mini"

# Hugginface model
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_API_KEY:
    HF_API_KEY = st.text_input("HF_API_TOKEN", value=None, type="password")

llm = InferenceClient(
	provider="together",
	api_key=HF_API_KEY
)
modelo = "deepseek-ai/DeepSeek-R1"

# Hugginface model
# HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# print(HF_API_KEY)
# llm = InferenceClient(
# 	provider="sambanova",
# 	api_key=HF_API_KEY
# )
# modelo = "meta-llama/Llama-3.3-70B-Instruct"

# Streamlit UI
st.title(f"Chat with {modelo}")

# Initialize session state for chat history if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("Type your message...")
if user_input:
    # Append user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # Construct the prompt with memory
    try:
        # Generate response from GPT-4o-mini
        response = llm.chat.completions.create(
                    model=modelo,
                    messages=st.session_state.chat_history, 
                    max_tokens=TOKENS_MAXIMOS
                    )
        #st.write(response)
        #bot_reply = response['choices'][0]["message"]["content"].strip()
        #bot_reply = response["choices"][0]["text"].strip()
        bot_reply = response.choices[0].message.content.strip()
        #st.write(bot_reply)
    except Exception as e:
        bot_reply = "I'm sorry, I'm not able to generate a response at this moment."
    
    # Append assistant message to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.write(bot_reply)