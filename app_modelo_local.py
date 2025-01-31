import streamlit as st
from llama_cpp import Llama

# Load the local LLaMA model
#MODEL_PATH ="../models/llama-2-7b.Q4_K_M.gguf"
MODEL_PATH="../models/deepseek-coder-6.7b-instruct.Q5_K_S.gguf"

llm = Llama(model_path=MODEL_PATH,                   
            chat_format="llama-2"
            )

# Streamlit UI
st.title("Chat with LLaMA")

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
    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
    prompt += "\nassistant: "
    
    # Generate response from LLaMA
    response = llm(prompt, 
                   max_tokens=2048, 
                   stop=["user:", "assistant:"]
                   )
    #st.write(prompt)
    bot_reply = response["choices"][0]["text"].strip()
    
    # Append assistant message to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.write(bot_reply)