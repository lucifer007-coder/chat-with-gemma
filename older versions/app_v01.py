import os
import json
import tempfile
from typing import List

import streamlit as st

from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# -----------------------
# Configuration / Helpers
# -----------------------
OLLAYAMA_DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:1b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Where to persist vectorstore and conversation history
DATA_DIR = "chat_data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
DOCS_META = os.path.join(DATA_DIR, "docs_meta.json")
HISTORY_DIR = os.path.join(DATA_DIR, "histories")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

def get_llm(model_name: str, temperature: float, base_url: str):
    """Create a new Ollama LLM wrapper instance"""
    return Ollama(model=model_name, base_url=base_url, temperature=temperature)

def load_or_create_index(embeddings: SentenceTransformerEmbeddings, index_path: str = INDEX_DIR):
    """Load existing FAISS index or create a new empty one"""
    if os.path.exists(index_path) and os.listdir(index_path):
        try:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"Existing index found but failed to load: {e}. Creating new index.")
    
    # Create empty FAISS index with a dummy document
    dummy_doc = Document(page_content="dummy", metadata={})
    index = FAISS.from_documents([dummy_doc], embeddings)
    # Remove the dummy document
    index.delete([0])
    return index

def save_docs_meta(docs_meta: List[dict]):
    """Save document metadata to JSON file"""
    try:
        with open(DOCS_META, "w", encoding="utf-8") as f:
            json.dump(docs_meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save document metadata: {e}")

def load_docs_meta():
    """Load document metadata from JSON file"""
    if os.path.exists(DOCS_META):
        try:
            with open(DOCS_META, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Failed to load document metadata: {e}")
            return []
    return []

def ingest_files_to_index(files, embeddings: SentenceTransformerEmbeddings, index: FAISS, persist: bool = True):
    """Ingest uploaded files into the FAISS index"""
    docs_meta = load_docs_meta()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    new_docs_meta = []
    
    for uploaded_file in files:
        fname = uploaded_file.name
        suffix = fname.lower().split(".")[-1]
        tmp_path = None
        
        try:
            if suffix in {"txt", "md"}:
                # Handle text files
                raw = uploaded_file.getvalue().decode("utf-8")
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}", mode='w', encoding='utf-8')
                tmp.write(raw)
                tmp.close()
                tmp_path = tmp.name
                loader = TextLoader(tmp_path, encoding="utf-8")
                docs = loader.load()
                
            elif suffix == "pdf":
                # Handle PDF files
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(uploaded_file.getvalue())
                tmp.close()
                tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
            else:
                st.warning(f"Unsupported file type: {fname}. Only txt, md, pdf are supported.")
                continue

            # Split documents into chunks
            chunks = text_splitter.split_documents(docs)
            
            # Add filename to metadata
            for chunk in chunks:
                chunk.metadata['source_file'] = fname
            
            # Add to index
            if chunks:
                index.add_documents(chunks)
                new_docs_meta.append({"filename": fname, "n_chunks": len(chunks)})
                
        except Exception as e:
            st.error(f"Error processing {fname}: {e}")
            
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # Persist index and metadata
    if persist and new_docs_meta:
        try:
            index.save_local(INDEX_DIR)
            all_meta = docs_meta + new_docs_meta
            save_docs_meta(all_meta)
        except Exception as e:
            st.error(f"Failed to persist index: {e}")
            
    return new_docs_meta

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Gemma Advanced Chat", layout="wide")
st.title("🤖 Gemma Chat (Ollama) — Advanced")

# Sidebar controls
st.sidebar.header("🔧 Model / Server")
base_url = st.sidebar.text_input("Ollama base URL", value=OLLAYAMA_DEFAULT_URL)
model_name = st.sidebar.text_input("Model", value=DEFAULT_MODEL, help="Local Ollama model to use")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

st.sidebar.header("💬 Mode")
mode = st.sidebar.radio("Choose chat mode", options=["Chat (memory)", "RAG (use docs)"])

st.sidebar.header("📝 System Prompt")
default_system = "You are a helpful assistant. Answer concisely and truthfully."
system_prompt = st.sidebar.text_area(
    "System message:", 
    value=default_system, 
    height=120,
    help="This message guides the assistant's behavior"
)

# Document management
st.sidebar.header("📚 Documents / RAG")
uploaded_files = st.sidebar.file_uploader(
    "Upload files to ingest (txt/md/pdf)", 
    accept_multiple_files=True,
    type=['txt', 'md', 'pdf']
)

if st.sidebar.button("📥 Ingest uploaded files"):
    if not uploaded_files:
        st.sidebar.warning("No files uploaded.")
    else:
        with st.spinner("Ingesting files..."):
            try:
                embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
                index = load_or_create_index(embeddings)
                new_meta = ingest_files_to_index(uploaded_files, embeddings, index)
                if new_meta:
                    st.sidebar.success(f"✅ Ingested {len(new_meta)} file(s).")
                else:
                    st.sidebar.info("No new chunks were ingested.")
            except Exception as e:
                st.sidebar.error(f"Failed to ingest files: {e}")

if st.sidebar.button("🗑️ Clear ingested documents"):
    try:
        # Remove index files
        if os.path.exists(INDEX_DIR):
            import shutil
            shutil.rmtree(INDEX_DIR)
        # Remove metadata
        if os.path.exists(DOCS_META):
            os.remove(DOCS_META)
        st.sidebar.success("✅ Cleared ingested documents.")
    except Exception as e:
        st.sidebar.error(f"Failed to clear documents: {e}")

# Show ingested documents
docs_meta = load_docs_meta()
if docs_meta:
    st.sidebar.markdown("**📄 Ingested docs:**")
    for m in docs_meta:
        st.sidebar.markdown(f"- {m.get('filename')} ({m.get('n_chunks')} chunks)")

# Conversation management
st.sidebar.header("💾 Conversation")
if st.sidebar.button("📥 Download conversation"):
    if "history" in st.session_state and st.session_state.history:
        conv_json = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
        st.sidebar.download_button(
            "Download JSON", 
            data=conv_json, 
            file_name="conversation.json", 
            mime="application/json"
        )
    else:
        st.sidebar.info("No conversation to download.")

uploaded_conv = st.sidebar.file_uploader("📤 Load conversation JSON", type="json")
if uploaded_conv is not None:
    try:
        payload = json.load(uploaded_conv)
        st.session_state.history = payload
        st.sidebar.success("✅ Conversation loaded.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to load conversation: {e}")

if st.sidebar.button("🗑️ Clear conversation"):
    st.session_state.history = []
    if "memory" in st.session_state:
        st.session_state.memory.clear()
    st.sidebar.success("✅ Cleared conversation.")
    st.rerun()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize LLM
llm_key = (model_name, temperature, base_url)
if "llm_cfg" not in st.session_state or st.session_state.get("llm_cfg") != llm_key:
    try:
        with st.spinner("Initializing LLM..."):
            st.session_state.llm = get_llm(model_name, temperature, base_url)
            st.session_state.llm_cfg = llm_key
        st.success("✅ LLM initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize Ollama LLM: {e}")
        st.info("Make sure Ollama is running and the model is available.")
        st.stop()

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 Chat")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display conversation history
        for entry in st.session_state.history:
            role = entry.get("role", "user")
            text = entry.get("text", "")
            
            if role == "user":
                with st.chat_message("user"):
                    st.write(text)
            else:
                with st.chat_message("assistant"):
                    st.write(text)

    # Chat input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Add user message to history
        st.session_state.history.append({"role": "user", "text": user_input})
        
        with st.spinner("Thinking..."):
            try:
                if mode == "Chat (memory)":
                    # Create memory with correct key for ConversationChain
                    memory = ConversationBufferMemory(memory_key="history", return_messages=False)
                    
                    # Add conversation history to memory
                    for entry in st.session_state.history[:-1]:  # Exclude the current user input
                        if entry["role"] == "user":
                            memory.chat_memory.add_user_message(entry["text"])
                        else:
                            memory.chat_memory.add_ai_message(entry["text"])
                    
                    # Create conversation chain
                    conv_chain = ConversationChain(
                        llm=st.session_state.llm, 
                        memory=memory,
                        verbose=False
                    )
                    
                    # Generate response with system prompt
                    full_input = f"{system_prompt}\n\nUser: {user_input}"
                    response = conv_chain.predict(input=full_input)
                    
                elif mode == "RAG (use docs)":
                    # RAG mode
                    if not docs_meta:
                        response = "⚠️ No documents have been ingested. Please upload documents first or switch to Chat mode."
                    else:
                        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
                        index = load_or_create_index(embeddings)
                        retriever = index.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                        
                        # Create memory for RAG chain (uses chat_history key)
                        rag_memory = ConversationBufferMemory(
                            memory_key="chat_history", 
                            return_messages=True,
                            output_key="answer"
                        )
                        
                        # Add conversation history to RAG memory
                        for entry in st.session_state.history[:-1]:  # Exclude current user input
                            if entry["role"] == "user":
                                rag_memory.chat_memory.add_user_message(entry["text"])
                            else:
                                rag_memory.chat_memory.add_ai_message(entry["text"])
                        
                        qa_chain = ConversationalRetrievalChain.from_llm(
                            llm=st.session_state.llm,
                            retriever=retriever,
                            memory=rag_memory,
                            verbose=False,
                            return_source_documents=True
                        )
                        
                        # Add system prompt to the question
                        enhanced_question = f"System context: {system_prompt}\n\nQuestion: {user_input}"
                        
                        result = qa_chain({
                            "question": enhanced_question,
                        })
                        
                        response = result["answer"]
                        
                        # Add sources information
                        if result.get("source_documents"):
                            sources = set()
                            for doc in result["source_documents"]:
                                if "source_file" in doc.metadata:
                                    sources.add(doc.metadata["source_file"])
                            if sources:
                                response += f"\n\n📚 *Sources: {', '.join(sources)}*"
                
                # Add assistant response to history
                st.session_state.history.append({"role": "assistant", "text": response})
                
                st.rerun()
                
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.session_state.history.append({"role": "assistant", "text": error_msg})
                st.error(error_msg)
                st.rerun()

with col2:
    st.subheader("🎛️ Controls")
    
    # Model info
    st.info(f"**Model:** {model_name}\n**Temperature:** {temperature}\n**Mode:** {mode}")
    
    # Save conversation
    if st.button("💾 Save conversation"):
        if st.session_state.history:
            try:
                os.makedirs(HISTORY_DIR, exist_ok=True)
                fname = os.path.join(HISTORY_DIR, f"conversation_{len(os.listdir(HISTORY_DIR))}.json")
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
                st.success(f"✅ Saved to {fname}")
            except Exception as e:
                st.error(f"Failed to save: {e}")
        else:
            st.info("No conversation to save.")

    # Debug info
    if st.button("🔍 Show debug info"):
        with st.expander("Debug Information"):
            st.write("**History length:**", len(st.session_state.history))
            st.write("**Last 3 messages:**")
            for entry in st.session_state.history[-3:]:
                st.write(f"- {entry['role']}: {entry['text'][:100]}...")

    # Tips
    with st.expander("💡 Tips"):
        st.markdown("""
        - **RAG mode**: Upload documents first for context-aware answers
        - **System prompt**: Guide assistant behavior and style  
        - **Temperature**: Lower = more focused, Higher = more creative
        - **Memory**: Conversation context is maintained automatically
        - **Sources**: RAG mode shows which documents were used
        """)

# Footer
st.markdown("---")
st.caption(f"🔗 Connected to: {base_url} | 🤖 Model: {model_name}")