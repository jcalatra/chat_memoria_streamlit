import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#st.write(OPENAI_API_KEY)

llm = OpenAI(api_key=OPENAI_API_KEY)


# Streamlit UI
st.title("Chat with Gpt-4o-mini")

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
                    model="gpt-4o-mini",
                    messages=st.session_state.chat_history, 
                    max_tokens=248
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