#
# Ideas:
#
# https://medium.com/predict/a-simple-comprehensive-guide-to-running-large-language-models-locally-on-cpu-and-or-gpu-using-c0c2a8483eee
#
# https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/
#
# https://docs.streamlit.io/library/api-reference/widgets/st.streamlit.chat_input
#

import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Load the local LLaMA model
#model_path ="../models/llama-2-7b.Q4_K_M.gguf"
#model_path="../models/deepseek-coder-6.7b-instruct.Q5_K_S.gguf"
# model_kwargs = {
#   "chat_format":"llama-2"
# }

# Streamlit UI
st.title("Chat with a local LLM")


# Another way to obtain the model

# List of models
nombres_modelos = [
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", 
    "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
    "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    ]
ficheros_modelos = [
    "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf", 
    "DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf",
    "mistral-7b-instruct-v0.1.Q5_K_M.gguf"
]

# Seleccionar modelo
seleccion_modelo = st.selectbox("Seleciona un modelo", 
                                nombres_modelos,
                                index=None,
                                placeholder="Select contact method...",)

if seleccion_modelo:
    st.write(f"Cargando el modelo seleccionado: {seleccion_modelo}")
        
    index_modelo = nombres_modelos.index(seleccion_modelo)
     
    # Download model from Hugging Face Hub
    model_path = hf_hub_download(nombres_modelos[index_modelo], 
                                 filename=ficheros_modelos[index_modelo])
    model_kwargs = {
    "n_ctx":4096,    # Context length to use
    "n_threads":4,   # Number of CPU threads to use
    "n_gpu_layers":0,# Number of model layers to offload to GPU. Set to 0 if only using CPU
    }

    ## Instantiate model from downloaded file
    llm = Llama(model_path=model_path, **model_kwargs)
    
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