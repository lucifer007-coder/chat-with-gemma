import os
import json
import tempfile
import asyncio
import time
import hashlib
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_ace import st_ace
import streamlit.components.v1 as components

# Core LangChain imports with updated imports
try:
    from langchain_ollama import OllamaLLM
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    from langchain.chains import ConversationChain
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_community.vectorstores.faiss import FAISS
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain_core.runnables.history import RunnableWithMessageHistory
except ImportError as e:
    st.error(f"LangChain import error: {e}")
    st.error("Please install required packages: pip install langchain-ollama langchain-community pypdf")
    st.stop()

# Additional imports with error handling
try:
    import nltk
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    st.error(f"Required package missing: {e}")
    st.stop()

# Download required NLTK data with better error handling
def download_nltk_data():
    """Download NLTK data with proper error handling"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            logging.warning(f"Failed to download NLTK punkt: {e}")
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logging.warning(f"Failed to download NLTK stopwords: {e}")

download_nltk_data()

# -----------------------
# Configuration & Constants
# -----------------------
OLLAMA_DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:1b"  # Fixed model name
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Directory structure
DATA_DIR = "chat_data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
HISTORY_DIR = os.path.join(DATA_DIR, "histories")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
DB_PATH = os.path.join(DATA_DIR, "chat_analytics.db")

# Create directories with error handling
def create_directories():
    """Create necessary directories with error handling"""
    for dir_path in [DATA_DIR, HISTORY_DIR, CACHE_DIR, LOGS_DIR]:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except PermissionError:
            st.error(f"Permission denied creating directory: {dir_path}")
            st.stop()
        except Exception as e:
            st.error(f"Error creating directory {dir_path}: {e}")
            st.stop()

create_directories()

# -----------------------
# Logging Setup
# -----------------------
def setup_logging():
    """Setup logging with error handling"""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(LOGS_DIR, 'chat_app.log')),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    except Exception as e:
        print(f"Logging setup failed: {e}")
        return logging.getLogger(__name__)

logger = setup_logging()

# -----------------------
# Data Models
# -----------------------
@dataclass
class ChatMessage:
    id: str
    role: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]
    confidence_score: Optional[float] = None
    sources: Optional[List[str]] = None
    response_time: Optional[float] = None

@dataclass
class DocumentMetadata:
    filename: str
    file_type: str
    size: int
    upload_time: datetime
    chunks_count: int
    checksum: str

# -----------------------
# Modern UI Components
# -----------------------
class ModernUI:
    @staticmethod
    def apply_custom_css():
        """Apply modern CSS styling"""
        st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Root variables for theming */
        :root {
            --primary-color: #6366f1;
            --primary-hover: #5855eb;
            --secondary-color: #f1f5f9;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
            --background-light: #f8fafc;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
            --radius-sm: 0.375rem;
            --radius-md: 0.5rem;
            --radius-lg: 0.75rem;
        }
        
        /* Global styles */
        .main {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom header */
        .app-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem 2rem;
            border-radius: var(--radius-lg);
            margin-bottom: 2rem;
            color: white;
            box-shadow: var(--shadow-lg);
        }
        
        .app-title {
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .app-subtitle {
            font-size: 1rem;
            opacity: 0.9;
            margin: 0.5rem 0 0 0;
            font-weight: 400;
        }
        
        /* Control panel */
        .control-panel {
            background: white;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-sm);
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: var(--radius-md);
            font-size: 0.875rem;
            font-weight: 500;
            margin-top: 1rem;
        }
        
        .status-online {
            background: #dcfce7;
            color: #166534;
        }
        
        .status-offline {
            background: #fee2e2;
            color: #991b1b;
        }
        
        /* Confidence badges */
        .confidence-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.25rem 0.5rem;
            border-radius: var(--radius-sm);
            font-size: 0.7rem;
            font-weight: 500;
        }
        
        .confidence-high { background: #dcfce7; color: #166534; }
        .confidence-medium { background: #fef3c7; color: #92400e; }
        .confidence-low { background: #fee2e2; color: #991b1b; }
        
        /* Metric cards */
        .metric-card {
            background: white;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            text-align: center;
            box-shadow: var(--shadow-sm);
            transition: transform 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary-color);
            margin: 0;
        }
        
        .metric-label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin: 0.5rem 0 0 0;
            font-weight: 500;
        }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .app-header {
                padding: 1rem;
                margin-bottom: 1rem;
            }
            
            .app-title {
                font-size: 1.5rem;
            }
            
            .control-panel {
                padding: 1rem;
            }
        }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_header():
        """Render modern app header"""
        st.markdown("""
        <div class="app-header">
            <h1 class="app-title">
                🤖 Chat With Gemma
            </h1>
            <p class="app-subtitle">
                Intelligent Conversations Powered by Google's Gemma Models
            </p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_status_indicator(is_online: bool, model_name: str):
        """Render connection status"""
        status_class = "status-online" if is_online else "status-offline"
        status_text = "Connected" if is_online else "Disconnected"
        icon = "🟢" if is_online else "🔴"
        
        st.markdown(f"""
        <div class="status-indicator {status_class}">
            {icon} {status_text} • {model_name}
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_metric_card(value: str, label: str, icon: str = "📊"):
        """Render a metric card"""
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_confidence_badge(score: float):
        """Render confidence score badge"""
        if score >= 0.7:
            badge_class = "confidence-high"
            icon = "🟢"
        elif score >= 0.4:
            badge_class = "confidence-medium"
            icon = "🟡"
        else:
            badge_class = "confidence-low"
            icon = "🔴"
        
        return f"""
        <span class="confidence-badge {badge_class}">
            {icon} {score:.2f}
        </span>
        """

# -----------------------
# Database Manager
# -----------------------
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT,
                    last_updated TEXT,
                    message_count INTEGER DEFAULT 0
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT,
                    response_time REAL,
                    confidence_score REAL,
                    sources TEXT,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    event_data TEXT,
                    timestamp TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
        except Exception as e:
            logger.error(f"Unexpected database error: {e}")
    
    def save_message(self, message: ChatMessage, conversation_id: str):
        """Save message to database with error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO messages 
                (id, conversation_id, role, content, timestamp, response_time, 
                 confidence_score, sources, metadata) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (message.id, conversation_id, message.role, message.content,
                  message.timestamp.isoformat(), message.response_time, message.confidence_score,
                  json.dumps(message.sources) if message.sources else None,
                  json.dumps(message.metadata)))
            
            conn.commit()
            conn.close()
            
        except sqlite3.Error as e:
            logger.error(f"Database save error: {e}")
        except Exception as e:
            logger.error(f"Unexpected save error: {e}")
    
    def get_analytics_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get analytics summary with error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total messages
            cursor.execute('''
                SELECT COUNT(*) FROM messages 
                WHERE timestamp >= datetime('now', '-{} days')
            '''.format(days))
            result = cursor.fetchone()
            total_messages = result[0] if result else 0
            
            # Average response time
            cursor.execute('''
                SELECT AVG(response_time) FROM messages 
                WHERE response_time IS NOT NULL 
                AND timestamp >= datetime('now', '-{} days')
            '''.format(days))
            result = cursor.fetchone()
            avg_response_time = result[0] if result and result[0] else 0
            
            # Average confidence
            cursor.execute('''
                SELECT AVG(confidence_score) FROM messages 
                WHERE confidence_score IS NOT NULL 
                AND timestamp >= datetime('now', '-{} days')
            '''.format(days))
            result = cursor.fetchone()
            avg_confidence = result[0] if result and result[0] else 0
            
            conn.close()
            
            return {
                'total_messages': total_messages,
                'avg_response_time': avg_response_time,
                'avg_confidence': avg_confidence
            }
            
        except sqlite3.Error as e:
            logger.error(f"Analytics query error: {e}")
            return {'total_messages': 0, 'avg_response_time': 0, 'avg_confidence': 0}
        except Exception as e:
            logger.error(f"Unexpected analytics error: {e}")
            return {'total_messages': 0, 'avg_response_time': 0, 'avg_confidence': 0}

# -----------------------
# Document Processor
# -----------------------
class DocumentProcessor:
    def __init__(self):
        try:
            self.embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            self.embeddings = None
    
    def process_document(self, file_content: bytes, filename: str) -> Tuple[List[Document], DocumentMetadata]:
        """Process uploaded document with comprehensive error handling"""
        if not self.embeddings:
            raise ValueError("Embeddings not initialized")
            
        file_type = filename.lower().split('.')[-1] if '.' in filename else 'unknown'
        checksum = hashlib.md5(file_content).hexdigest()
        
        # Save to temporary file
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
                tmp_file.write(file_content)
                tmp_file.flush()
                temp_file = tmp_file.name
                
                if file_type in ['txt', 'md']:
                    loader = TextLoader(temp_file, encoding='utf-8')
                elif file_type == 'pdf':
                    try:
                        loader = PyPDFLoader(temp_file)
                    except Exception as e:
                        raise ValueError(f"PDF processing failed. Please install pypdf: pip install pypdf")
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")
                
                docs = loader.load()
                
        except UnicodeDecodeError:
            # Try different encodings
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}', mode='w', encoding='latin-1') as tmp_file:
                    tmp_file.write(file_content.decode('latin-1'))
                    tmp_file.flush()
                    temp_file = tmp_file.name
                    loader = TextLoader(temp_file, encoding='latin-1')
                    docs = loader.load()
            except Exception as e:
                raise ValueError(f"Failed to decode file {filename}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load document {filename}: {e}")
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")
        
        if not docs:
            raise ValueError(f"No content extracted from {filename}")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        try:
            chunks = text_splitter.split_documents(docs)
        except Exception as e:
            raise ValueError(f"Failed to split document {filename}: {e}")
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'source_file': filename,
                'chunk_id': i,
                'file_type': file_type,
                'checksum': checksum
            })
        
        metadata = DocumentMetadata(
            filename=filename,
            file_type=file_type,
            size=len(file_content),
            upload_time=datetime.now(),
            chunks_count=len(chunks),
            checksum=checksum
        )
        
        return chunks, metadata

# -----------------------
# Chat Engine
# -----------------------
class ChatEngine:
    def __init__(self):
        self.db_manager = DatabaseManager(DB_PATH)
        self.doc_processor = DocumentProcessor()
        self.memory = ConversationBufferMemory(return_messages=True)
    
    def get_llm(self, model_name: str, temperature: float, base_url: str):
        """Create LLM instance with error handling"""
        try:
            return OllamaLLM(
                model=model_name,
                base_url=base_url,
                temperature=temperature,
                timeout=30
            )
        except Exception as e:
            logger.error(f"Failed to create LLM: {e}")
            raise
    
    def test_connection(self, model_name: str, base_url: str) -> bool:
        """Test Ollama connection with proper error handling"""
        try:
            llm = OllamaLLM(model=model_name, base_url=base_url, timeout=10)
            # Test with a simple prompt
            response = llm.invoke("Hello")
            return True
        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
            return False
    
    def calculate_confidence_score(self, response: str, sources: List[str] = None) -> float:
        """Calculate response confidence score"""
        if not response or not response.strip():
            return 0.0
            
        score = 0.5
        
        # Length factor
        if len(response) > 50:
            score += 0.1
        
        # Sources factor
        if sources and len(sources) > 0:
            score += 0.2
        
        # Check for uncertainty indicators
        uncertainty_words = ['maybe', 'might', 'possibly', 'not sure', 'i think', 'perhaps']
        uncertainty_count = sum(1 for word in uncertainty_words if word in response.lower())
        score -= uncertainty_count * 0.1
        
        # Check for confidence indicators
        confidence_words = ['definitely', 'certainly', 'clearly', 'obviously']
        confidence_count = sum(1 for word in confidence_words if word in response.lower())
        score += confidence_count * 0.05
        
        return max(0.0, min(1.0, score))
    
    def generate_response(self, user_input: str, mode: str, system_prompt: str, 
                         model_config: Dict[str, Any]) -> ChatMessage:
        """Generate response based on mode with comprehensive error handling"""
        start_time = time.time()
        
        try:
            llm = self.get_llm(**model_config)
            
            if mode == "Chat":
                response = self._generate_chat_response(user_input, system_prompt, llm)
                sources = None
            else:  # RAG mode
                response, sources = self._generate_rag_response(user_input, system_prompt, llm)
            
            response_time = time.time() - start_time
            confidence_score = self.calculate_confidence_score(response, sources)
            
            return ChatMessage(
                id=str(uuid.uuid4()),
                role="assistant",
                content=response,
                timestamp=datetime.now(),
                metadata={
                    "mode": mode,
                    "model": model_config["model_name"],
                    "temperature": model_config["temperature"]
                },
                confidence_score=confidence_score,
                sources=sources,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ChatMessage(
                id=str(uuid.uuid4()),
                role="assistant",
                content=f"❌ Error: {str(e)}",
                timestamp=datetime.now(),
                metadata={"error": True},
                confidence_score=0.0,
                response_time=time.time() - start_time
            )
    
    def _generate_chat_response(self, user_input: str, system_prompt: str, llm) -> str:
        """Generate chat response with memory and error handling"""
        try:
            # Simple approach without deprecated ConversationChain
            if system_prompt and system_prompt.strip():
                full_prompt = f"System: {system_prompt}\n\nUser: {user_input}"
            else:
                full_prompt = user_input
                
            response = llm.invoke(full_prompt)
            return response
            
        except Exception as e:
            logger.error(f"Chat response error: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _generate_rag_response(self, user_input: str, system_prompt: str, llm) -> Tuple[str, List[str]]:
        """Generate RAG response using documents with error handling"""
        try:
            if not st.session_state.get('faiss_index') or not st.session_state.get('documents'):
                return "⚠️ No documents available. Please upload documents first.", []
            
            # Retrieve relevant documents
            try:
                relevant_docs = st.session_state.faiss_index.similarity_search(user_input, k=3)
            except Exception as e:
                logger.error(f"Document retrieval error: {e}")
                return f"❌ Error retrieving documents: {str(e)}", []
            
            if not relevant_docs:
                return "❌ No relevant information found in documents.", []
            
            # Create context
            context_parts = []
            sources = []
            
            for doc in relevant_docs:
                if doc.page_content and doc.page_content.strip():
                    context_parts.append(doc.page_content.strip())
                    source_file = doc.metadata.get('source_file', 'Unknown')
                    if source_file not in sources:
                        sources.append(source_file)
            
            if not context_parts:
                return "❌ No valid content found in retrieved documents.", []
            
            context = "\n\n".join(context_parts)
            
            # Generate response
            prompt_parts = []
            if system_prompt and system_prompt.strip():
                prompt_parts.append(f"System: {system_prompt}")
            
            prompt_parts.extend([
                f"Context: {context}",
                f"Question: {user_input}",
                "Answer based on the context provided:"
            ])
            
            prompt = "\n\n".join(prompt_parts)
            
            try:
                response = llm.invoke(prompt)
            except Exception as e:
                logger.error(f"LLM invocation error: {e}")
                return f"❌ Error generating response: {str(e)}", sources
            
            if sources:
                response += f"\n\n📚 **Sources:** {', '.join(sources)}"
            
            return response, sources
            
        except Exception as e:
            logger.error(f"RAG response error: {e}")
            return f"❌ Error in RAG processing: {str(e)}", []

# -----------------------
# Main Application
# -----------------------
class AdvancedChatApp:
    def __init__(self):
        self.ui = ModernUI()
        self.chat_engine = ChatEngine()
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state with proper defaults"""
        defaults = {
            "conversation_id": str(uuid.uuid4()),
            "messages": [],
            "documents": [],
            "faiss_index": None,
            "current_tab": "Chat",
            "system_prompt": "You are a helpful AI assistant. Provide accurate, concise, and helpful responses."
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def test_ollama_connection(self, base_url: str, model_name: str) -> bool:
        """Test Ollama connection with caching"""
        cache_key = f"connection_{base_url}_{model_name}"
        
        # Check cache (valid for 30 seconds)
        if hasattr(st.session_state, cache_key):
            cached_time, cached_result = st.session_state[cache_key]
            if time.time() - cached_time < 30:
                return cached_result
        
        # Test connection
        result = self.chat_engine.test_connection(model_name, base_url)
        
        # Cache result
        st.session_state[cache_key] = (time.time(), result)
        
        return result
    
    def render_control_panel(self):
        """Render the main control panel"""
        with st.container():
            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            
            with col1:
                mode = st.selectbox(
                    "💬 Mode",
                    ["Chat", "RAG"],
                    help="Chat: Conversation mode | RAG: Document Q&A",
                    key="mode_select"
                )
            
            with col2:
                model_name = st.selectbox(
                    "🤖 Model",
                    [DEFAULT_MODEL, "gemma3:270m"],
                    help="Select AI model",
                    key="model_select"
                )
            
            with col3:
                temperature = st.slider(
                    "🌡️ Temperature",
                    0.0, 1.0, 0.3, 0.1,
                    help="Creativity level",
                    key="temperature_slider"
                )
            
            with col4:
                base_url = st.text_input(
                    "🔗 Ollama URL",
                    value=OLLAMA_DEFAULT_URL,
                    help="Ollama server URL",
                    key="base_url_input"
                )
            
            # Connection status
            is_connected = self.test_ollama_connection(base_url, model_name)
            self.ui.render_status_indicator(is_connected, model_name)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            return {
                "mode": mode,
                "model_name": model_name,
                "temperature": temperature,
                "base_url": base_url,
                "is_connected": is_connected
            }
    
    def render_system_prompt_section(self):
        """Render system prompt configuration"""
        with st.expander("⚙️ System Configuration", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                system_prompt = st.text_area(
                    "System Prompt",
                    value=st.session_state.get("system_prompt", ""),
                    height=100,
                    help="Instructions that guide the AI's behavior",
                    key="system_prompt_area"
                )
                st.session_state.system_prompt = system_prompt
            
            with col2:
                st.write("**Quick Presets:**")
                
                presets = {
                    "💼 Professional": "You are a professional assistant. Provide formal, accurate responses with proper business etiquette.",
                    "🎓 Academic": "You are an academic assistant. Provide detailed, well-researched responses with citations when possible.",
                    "💡 Creative": "You are a creative assistant. Provide imaginative and innovative responses while maintaining accuracy."
                }
                
                for label, prompt in presets.items():
                    if st.button(label, use_container_width=True, key=f"preset_{label}"):
                        st.session_state.system_prompt = prompt
                        st.rerun()
        
        return st.session_state.system_prompt
    
    def render_document_management(self):
        """Render document management interface"""
        st.subheader("📚 Document Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Upload Documents",
                accept_multiple_files=True,
                type=['txt', 'md', 'pdf'],
                help="Upload documents for RAG mode",
                key="document_uploader"
            )
        
        with col2:
            if st.session_state.documents:
                st.success(f"📄 {len(st.session_state.documents)} documents loaded")
                if st.button("🗑️ Clear All", use_container_width=True, key="clear_docs"):
                    self.clear_documents()
                    st.rerun()
            else:
                st.info("No documents loaded")
        
        if uploaded_files:
            if st.button("📥 Process Documents", type="primary", use_container_width=True, key="process_docs"):
                self.process_documents(uploaded_files)
                st.rerun()
        
        # Display document list
        if st.session_state.documents:
            st.write("**Loaded Documents:**")
            doc_files = list(set([doc.metadata.get('source_file', 'Unknown') for doc in st.session_state.documents]))
            for i, filename in enumerate(doc_files, 1):
                st.write(f"{i}. {filename}")
    
    def process_documents(self, uploaded_files):
        """Process uploaded documents with comprehensive error handling"""
        if not uploaded_files:
            st.warning("No files selected")
            return
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_chunks = []
        processed_files = 0
        failed_files = []
        
        for i, file in enumerate(uploaded_files):
            try:
                status_text.text(f"Processing {file.name}...")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                if file.size == 0:
                    failed_files.append(f"{file.name} (empty file)")
                    continue
                
                if file.size > 10 * 1024 * 1024:  # 10MB limit
                    failed_files.append(f"{file.name} (file too large)")
                    continue
                
                chunks, metadata = self.chat_engine.doc_processor.process_document(
                    file.getvalue(), file.name
                )
                
                if chunks:
                    all_chunks.extend(chunks)
                    processed_files += 1
                else:
                    failed_files.append(f"{file.name} (no content extracted)")
                
            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")
                failed_files.append(f"{file.name} ({str(e)})")
        
        # Create or update FAISS index
        if all_chunks:
            try:
                status_text.text("Creating search index...")
                embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
                st.session_state.faiss_index = FAISS.from_documents(all_chunks, embeddings)
                st.session_state.documents = all_chunks
                
                success_msg = f"✅ Successfully processed {processed_files} documents with {len(all_chunks)} chunks!"
                if failed_files:
                    success_msg += f"\n\n⚠️ Failed to process {len(failed_files)} files:\n" + "\n".join(failed_files)
                
                st.success(success_msg)
                
            except Exception as e:
                logger.error(f"Error creating FAISS index: {e}")
                st.error(f"❌ Error creating search index: {e}")
        else:
            error_msg = "❌ No documents were successfully processed."
            if failed_files:
                error_msg += f"\n\nFailed files:\n" + "\n".join(failed_files)
            st.error(error_msg)
        
        progress_bar.empty()
        status_text.empty()
    
    def clear_documents(self):
        """Clear all documents and reset index"""
        st.session_state.documents = []
        st.session_state.faiss_index = None
        st.toast("Documents cleared! 🗑️")
    
    def render_chat_interface(self, config: Dict[str, Any]):
        """Render the chat interface"""
        # Display messages
        for message in st.session_state.messages:
            self.render_message(message)
        
        # Chat input
        user_input = st.chat_input(
            "Type your message here...", 
            disabled=not config["is_connected"],
            key="chat_input"
        )
        
        if user_input and config["is_connected"]:
            # Add user message
            user_message = ChatMessage(
                id=str(uuid.uuid4()),
                role="user",
                content=user_input,
                timestamp=datetime.now(),
                metadata={}
            )
            st.session_state.messages.append(user_message)
            
            # Generate AI response
            with st.spinner("🤔 Thinking..."):
                ai_message = self.chat_engine.generate_response(
                    user_input=user_input,
                    mode=config["mode"],
                    system_prompt=config["system_prompt"],
                    model_config={
                        "model_name": config["model_name"],
                        "temperature": config["temperature"],
                        "base_url": config["base_url"]
                    }
                )
                
                st.session_state.messages.append(ai_message)
                
                # Save to database
                try:
                    self.chat_engine.db_manager.save_message(user_message, st.session_state.conversation_id)
                    self.chat_engine.db_manager.save_message(ai_message, st.session_state.conversation_id)
                except Exception as e:
                    logger.error(f"Failed to save messages: {e}")
            
            st.rerun()
        elif user_input and not config["is_connected"]:
            st.error("❌ Please check your Ollama connection before sending messages.")
    
    def render_message(self, message: ChatMessage):
        """Render a single message with enhanced UI"""
        if message.role == "user":
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)
                
                # Message metadata
                if message.confidence_score is not None or message.response_time is not None:
                    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                    
                    with col1:
                        if message.confidence_score is not None:
                            confidence_html = self.ui.render_confidence_badge(message.confidence_score)
                            st.markdown(confidence_html, unsafe_allow_html=True)
                    
                    with col2:
                        if message.response_time is not None:
                            st.caption(f"⏱️ {message.response_time:.1f}s")
                    
                    with col3:
                        if st.button("👍", key=f"like_{message.id}", help="Good response"):
                            st.toast("Thanks for the feedback! 👍")
                    
                    with col4:
                        if st.button("👎", key=f"dislike_{message.id}", help="Poor response"):
                            st.toast("Thanks for the feedback! We'll improve.")
    
    def render_analytics_tab(self):
        """Render analytics dashboard"""
        st.subheader("📊 Analytics Dashboard")
        
        # Get analytics data
        analytics = self.chat_engine.db_manager.get_analytics_summary()
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.ui.render_metric_card(
                str(analytics['total_messages']), 
                "Total Messages", 
                "💬"
            )
        
        with col2:
            self.ui.render_metric_card(
                f"{analytics['avg_response_time']:.1f}s", 
                "Avg Response Time", 
                "⏱️"
            )
        
        with col3:
            self.ui.render_metric_card(
                f"{analytics['avg_confidence']:.2f}", 
                "Avg Confidence", 
                "🎯"
            )
        
        with col4:
            self.ui.render_metric_card(
                str(len(st.session_state.documents)), 
                "Documents", 
                "📚"
            )
        
        # Additional analytics
        if analytics['total_messages'] > 0:
            st.subheader("📈 Conversation Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("**Recent Activity**\n\nYour chat activity has been recorded. More detailed analytics will be available in future updates.")
            
            with col2:
                st.info("**Performance Metrics**\n\nResponse times and confidence scores are being tracked to improve the chat experience.")
        else:
            st.info("💬 Start chatting to see analytics!")
    
    def render_settings_tab(self):
        """Render settings and configuration"""
        st.subheader("⚙️ Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Chat Settings**")
            
            auto_save = st.checkbox("Auto-save conversations", value=True, key="auto_save_setting")
            show_timestamps = st.checkbox("Show message timestamps", value=False, key="timestamps_setting")
            compact_mode = st.checkbox("Compact message display", value=False, key="compact_setting")
            
            st.write("**Export Options**")
            if st.button("📥 Export Chat History", use_container_width=True, key="export_chat"):
                self.export_chat_history()
        
        with col2:
            st.write("**System Information**")
            
            st.info(f"""
            **Version:** 2.0.0  
            **Documents:** {len(st.session_state.documents)}  
            **Messages:** {len(st.session_state.messages)}  
            **Conversation ID:** {st.session_state.conversation_id[:8]}...
            """)
            
            st.write("**Reset Options**")
            if st.button("🔄 Clear Chat History", use_container_width=True, key="clear_chat"):
                st.session_state.messages = []
                st.session_state.conversation_id = str(uuid.uuid4())
                self.chat_engine.memory.clear()
                st.toast("Chat history cleared! 🔄")
                st.rerun()
            
            if st.button("⚠️ Reset All Data", use_container_width=True, key="reset_all"):
                if st.button("🚨 Confirm Reset", type="secondary", key="confirm_reset"):
                    self.reset_application()
                    st.rerun()
    
    def export_chat_history(self):
        """Export chat history as JSON"""
        if st.session_state.messages:
            try:
                chat_data = {
                    'conversation_id': st.session_state.conversation_id,
                    'messages': [asdict(msg) for msg in st.session_state.messages],
                    'exported_at': datetime.now().isoformat(),
                    'total_messages': len(st.session_state.messages)
                }
                
                json_str = json.dumps(chat_data, indent=2, default=str)
                
                st.download_button(
                    "📥 Download JSON",
                    data=json_str,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_json"
                )
                
                st.success("✅ Chat history ready for download!")
                
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
        else:
            st.warning("No messages to export!")
    
    def reset_application(self):
        """Reset the entire application state"""
        # Clear session state
        keys_to_keep = ['system_prompt']  # Keep some settings
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        
        # Reinitialize
        self.init_session_state()
        self.chat_engine.memory.clear()
        
        st.toast("Application reset! 🔄")
    
    def run(self):
        """Run the main application"""
        st.set_page_config(
            page_title="Chat With Gemma",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Apply custom CSS
        self.ui.apply_custom_css()
        
        # Render header
        self.ui.render_header()
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📚 Documents", "📊 Analytics", "⚙️ Settings"])
        
        with tab1:
            # Control panel
            config = self.render_control_panel()
            
            # System prompt
            system_prompt = self.render_system_prompt_section()
            config["system_prompt"] = system_prompt
            
            # Quick actions
            col1, col2, col3 = st.columns([1, 1, 4])
            
            with col1:
                if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_main"):
                    st.session_state.messages = []
                    st.session_state.conversation_id = str(uuid.uuid4())
                    self.chat_engine.memory.clear()
                    st.rerun()
            
            with col2:
                if st.button("💾 Save Chat", use_container_width=True, key="save_chat_main"):
                    st.toast("Chat saved! 💾")
            
            # Chat interface
            self.render_chat_interface(config)
        
        with tab2:
            self.render_document_management()
        
        with tab3:
            self.render_analytics_tab()
        
        with tab4:
            self.render_settings_tab()

# -----------------------
# Main Entry Point
# -----------------------
def main():
    """Main application entry point with comprehensive error handling"""
    try:
        app = AdvancedChatApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        logger.error(f"Application error: {e}")
        
        with st.expander("🐛 Debug Information"):
            import traceback
            st.code(traceback.format_exc())
        
        st.info("💡 Try refreshing the page or check the logs for more details.")

if __name__ == "__main__":
    main()