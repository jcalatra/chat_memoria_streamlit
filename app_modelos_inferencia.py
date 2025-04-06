#
# Ideas:
#
# https://medium.com/predict/a-simple-comprehensive-guide-to-running-large-language-models-locally-on-cpu-and-or-gpu-using-c0c2a8483eee
#
# https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/
#
# https://docs.streamlit.io/library/api-reference/widgets/st.streamlit.chat_input
#
# https://huggingface.co/docs/api-inference/tasks/chat-completion
#

import os
from dotenv import load_dotenv
import streamlit as st
#from llama_cpp import Llama
#from huggingface_hub import hf_hub_download
from huggingface_hub import InferenceClient



# Streamlit UI
st.title("Chat with a local LLM")

# List of models
nombres_modelos = [
    "google/gemma-2-2b-it",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    ]
proveedor_modelos = [
    "hf-inference",
    "hf-inference",
]

# Seleccionar modelo
seleccion_modelo = st.selectbox("Seleciona un modelo", 
                                nombres_modelos,
                                index=None,
                                placeholder="Select contact method...",)

if seleccion_modelo:
    st.write(f"Cargando el modelo seleccionado: {seleccion_modelo}")
        
    index_modelo = nombres_modelos.index(seleccion_modelo)
    
    load_dotenv()
    HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not HF_API_TOKEN:
        HF_API_TOKEN = st.text_input("HF_API_TOKEN", value=None, type="password")
        
    if HF_API_TOKEN:
        ## Instantiate model from downloaded file    
        client = InferenceClient(provider=proveedor_modelos[index_modelo], 
                                 api_key=HF_API_TOKEN)
        
        # Inicio del chat 
        st.header("Pregúntale al modelo")

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
            # prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
            # prompt += "\nassistant: "
            prompt = []
            for msg in st.session_state.chat_history:
                prompt.append({'role': msg['role'], 'content': msg['content']})
            
            #st.write(prompt)
            
            # Generate response
            # response = client.text_generation(
            #         model=nombres_modelos[index_modelo],
            #         inputs=prompt,
            #         provider=proveedor_modelos[index_modelo],
            #         )
            #bot_reply = response["choices"][0]["text"].strip()

            completion = client.chat.completions.create(
                        model=nombres_modelos[index_modelo], 
                        messages=prompt, 
                        # messages= [ {   "role": "user",
                        #                 "content": "What is the capital of France?" } ], 
                        max_tokens=2500,
                    )
            bot_reply = completion.choices[0].message.content
            #st.write(completion)
            
            # Append assistant message to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
            with st.chat_message("assistant"):
                st.write(bot_reply)