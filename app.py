# app.py
import os
import streamlit as st
import qdrant_client
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import PromptTemplate, Settings

# Initialize Qdrant client
@st.cache_resource
def init_qdrant():
    return qdrant_client.QdrantClient(
        host="localhost",
        port=6333
    )

# Initialize embedding model
@st.cache_resource
def init_embedding():
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5",
        trust_remote_code=True
    )

# Initialize LLM based on selection
def init_llm(model_choice):
    st.sidebar.write(f"Initializing {model_choice}...")
    if model_choice == "OpenAI GPT-3.5":
        llm = OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=512
        )
        st.sidebar.write("OpenAI Settings:", {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 512
        })
        return llm
    elif model_choice == "Llama 3.2":
        llm = Ollama(
            model="llama3.2:1b",
            request_timeout=120.0
        )
        st.sidebar.write("Llama Settings:", {
            "model": "llama3.2:1b",
            "request_timeout": 120.0
        })
        return llm
    else:  # DeepSeek
        llm = Ollama(
            model="deepseek-coder:6.7b",
            request_timeout=120.0
        )
        st.sidebar.write("DeepSeek Settings:", {
            "model": "deepseek-coder:6.7b",
            "request_timeout": 120.0
        })
        return llm

# Initialize query engine
@st.cache_resource(show_spinner=False)
def init_query_engine(_client, collection_name, _llm):
    try:
        # Set up embedding model
        embed_model = init_embedding()
        
        # Set up vector store
        vector_store = QdrantVectorStore(
            client=_client, 
            collection_name=collection_name
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # Create index
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        # Create reranker
        rerank = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2", 
            top_n=3
        )
        
        # Define prompt template
        template = """Context information is below:
                    ---------------------
                    {context_str}
                    ---------------------
                    Based on the context above, analyze the query and provide the response in the following format:
                    
                    Scenario: [Describe the situation from matching context]
                    Remediation: [Provide specific prevention/remediation steps]
                    Points of contact: [List relevant contact information/helplines]
                    
                    If no relevant information is found in the context, respond with "No matching scenario found."
                    
                    Query: {query_str}
                    
                    Response:"""
        
        qa_prompt_tmpl = PromptTemplate(template)
        
        # Set the LLM in Settings
        Settings.llm = _llm
        
        # Create and configure query engine
        query_engine = index.as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[rerank]
        )
        
        # Update query engine with template
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
        )
        
        return query_engine
    except Exception as e:
        st.error(f"Error initializing query engine: {str(e)}")
        raise e

def main():
    st.title("Cyber Security RAG Chatbot")
    
    # Add model selection dropdown in the sidebar
    model_choice = st.sidebar.selectbox(
        "Select LLM Model",
        ["OpenAI GPT-3.5", "Llama 3.2", "DeepSeek"]
    )
    
    # Handle API key based on model selection
    if model_choice == "OpenAI GPT-3.5" and 'OPENAI_API_KEY' not in st.session_state:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state['OPENAI_API_KEY'] = api_key
            st.success("API key set successfully!")
    
    # Check if OpenAI key is needed and present
    proceed = True
    if model_choice == "OpenAI GPT-3.5" and 'OPENAI_API_KEY' not in st.session_state:
        proceed = False
    
    if proceed:
        try:
            # Initialize components
            client = init_qdrant()
            collection_name = "demo_29thJan"
            
            # Initialize LLM based on selection
            llm = init_llm(model_choice)
            
            # Initialize query engine with selected LLM
            query_engine = init_query_engine(client, collection_name, llm)
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("What would you like to know about cyber security?"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner(f"Thinking using {model_choice}..."):
                        response = query_engine.query(prompt)
                        # Add model info to the response
                        response_with_model = f"[Response from {model_choice}]\n\n{str(response)}"
                        st.markdown(response_with_model)
                        st.session_state.messages.append({"role": "assistant", "content": response_with_model})
                        
                        # Log model details for verification
                        st.sidebar.write(f"Last response generated using: {model_choice}")
                        if model_choice == "Llama 3.2":
                            st.sidebar.write("Ollama Model Settings:", {
                                "model": "llama3.2:1b",
                                "request_timeout": 120.0
                            })
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()