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
    from langchain_ollama import OllamaLLM, ChatOllama
    from langchain.memory import ConversationBufferMemory
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_community.vectorstores.faiss import FAISS
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain_community.tools import DuckDuckGoSearchRun
except ImportError as e:
    st.error(f"LangChain import error: {e}")
    st.error("Please install required packages: pip install langchain-ollama langchain-community pypdf duckduckgo-search")
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

# OCR-related imports
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("OCR packages not found. OCR functionality disabled.")

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
DEFAULT_MODEL = "gemma3n:e4b"
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
    for dir_path in [DATA_DIR, HISTORY_DIR, CACHE_DIR, LOGS_DIR, INDEX_DIR]:
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
    feedback: Optional[int] = None  # 1 for like, -1 for dislike

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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
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
        .main {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
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
        .control-panel {
            background: white;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-sm);
        }
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
        .typing-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            color: var(--text-secondary);
        }
        .typing-dot {
            width: 6px;
            height: 6px;
            background: var(--text-secondary);
            border-radius: 50%;
            animation: blink 1.4s infinite both;
        }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes blink {
            0%, 80%, 100% { opacity: 0; }
            40% { opacity: 1; }
        }
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

    @staticmethod
    def render_typing_indicator():
        """Render typing indicator"""
        st.markdown("""
        <div class="typing-indicator">
            Typing <span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>
        </div>
        """, unsafe_allow_html=True)

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
            
            # Create conversations table with summary
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    summary TEXT,
                    created_at TEXT,
                    last_updated TEXT,
                    message_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create messages table with feedback
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
                    feedback INTEGER,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            ''')
            
            # Create analytics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    event_data TEXT,
                    timestamp TEXT
                )
            ''')
            
            # Migrate existing database if needed
            cursor.execute("PRAGMA table_info(conversations)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'summary' not in columns:
                cursor.execute("ALTER TABLE conversations ADD COLUMN summary TEXT")
            
            cursor.execute("PRAGMA table_info(messages)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'feedback' not in columns:
                cursor.execute("ALTER TABLE messages ADD COLUMN feedback INTEGER")
            
            conn.commit()
            conn.close()
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
        except Exception as e:
            logger.error(f"Unexpected database error: {e}")
    
    def save_conversation(self, conversation_id: str, title: str = None, summary: str = None):
        """Save or update conversation metadata"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT OR REPLACE INTO conversations 
                (id, title, summary, created_at, last_updated, message_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (conversation_id, title, summary, now, now, 0))
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Save conversation error: {e}")
    
    def update_conversation(self, conversation_id: str, title: Optional[str] = None, summary: Optional[str] = None, increment_count: bool = False):
        """Update conversation details"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if title is not None:
                updates.append("title = ?")
                params.append(title)
            
            if summary is not None:
                updates.append("summary = ?")
                params.append(summary)
            
            updates.append("last_updated = ?")
            params.append(datetime.now().isoformat())
            
            if increment_count:
                updates.append("message_count = message_count + 1")
            
            if updates:
                query = f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?"
                params.append(conversation_id)
                cursor.execute(query, params)
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Update conversation error: {e}")
    
    def get_conversations(self) -> List[Dict]:
        """Get all conversations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, title, created_at, message_count 
                FROM conversations 
                ORDER BY last_updated DESC
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'id': row[0],
                    'title': row[1] or f"Conversation {row[0][:8]}",
                    'created_at': row[2],
                    'message_count': row[3]
                } for row in rows
            ]
        except sqlite3.Error as e:
            logger.error(f"Get conversations error: {e}")
            return []
    
    def get_conversation_summary(self, conversation_id: str) -> str:
        """Get conversation summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT summary FROM conversations WHERE id = ?', (conversation_id,))
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
        except sqlite3.Error as e:
            logger.error(f"Get summary error: {e}")
            return None
    
    def load_messages(self, conversation_id: str) -> List[ChatMessage]:
        """Load messages for a conversation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, role, content, timestamp, response_time, confidence_score, 
                       sources, metadata, feedback
                FROM messages 
                WHERE conversation_id = ? 
                ORDER BY timestamp ASC
            ''', (conversation_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            messages = []
            for row in rows:
                sources = json.loads(row[6]) if row[6] else None
                metadata = json.loads(row[7]) if row[7] else {}
                messages.append(ChatMessage(
                    id=row[0],
                    role=row[1],
                    content=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    metadata=metadata,
                    confidence_score=row[5],
                    sources=sources,
                    response_time=row[4],
                    feedback=row[8]
                ))
            
            return messages
        except sqlite3.Error as e:
            logger.error(f"Load messages error: {e}")
            return []
    
    def save_message(self, message: ChatMessage, conversation_id: str):
        """Save message to database with error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO messages 
                (id, conversation_id, role, content, timestamp, response_time, 
                 confidence_score, sources, metadata, feedback) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (message.id, conversation_id, message.role, message.content,
                  message.timestamp.isoformat(), message.response_time, message.confidence_score,
                  json.dumps(message.sources) if message.sources else None,
                  json.dumps(message.metadata), message.feedback))
            
            conn.commit()
            conn.close()
            
            self.update_conversation(conversation_id, increment_count=True)
            
        except sqlite3.Error as e:
            logger.error(f"Database save error: {e}")
        except Exception as e:
            logger.error(f"Unexpected save error: {e}")
    
    def update_message_feedback(self, message_id: str, feedback: int):
        """Update message feedback"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE messages SET feedback = ? WHERE id = ?
            ''', (feedback, message_id))
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Update feedback error: {e}")
    
    def get_analytics_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get analytics summary with error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Total messages
            cursor.execute('''
                SELECT COUNT(*) FROM messages 
                WHERE timestamp >= ?
            ''', (since_date,))
            total_messages = cursor.fetchone()[0] or 0
            
            # User vs Assistant ratio
            cursor.execute('''
                SELECT role, COUNT(*) FROM messages 
                WHERE timestamp >= ?
                GROUP BY role
            ''', (since_date,))
            roles = dict(cursor.fetchall())
            user_count = roles.get('user', 0)
            assistant_count = roles.get('assistant', 0)
            ratio = f"{user_count}:{assistant_count}" if assistant_count else "N/A"
            
            # Average response time
            cursor.execute('''
                SELECT AVG(response_time) FROM messages 
                WHERE response_time IS NOT NULL AND timestamp >= ?
            ''', (since_date,))
            avg_response_time = cursor.fetchone()[0] or 0
            
            # Average confidence
            cursor.execute('''
                SELECT AVG(confidence_score) FROM messages 
                WHERE confidence_score IS NOT NULL AND timestamp >= ?
            ''', (since_date,))
            avg_confidence = cursor.fetchone()[0] or 0
            
            # Usage by mode
            cursor.execute('''
                SELECT metadata, COUNT(*) FROM messages 
                WHERE role = 'assistant' AND timestamp >= ?
                GROUP BY json_extract(metadata, '$.mode')
            ''', (since_date,))
            modes = dict((json.loads(row[0])['mode'], row[1]) for row in cursor.fetchall() if row[0])
            
            # Average tokens (assuming content length as proxy)
            cursor.execute('''
                SELECT AVG(LENGTH(content)) FROM messages 
                WHERE role = 'assistant' AND timestamp >= ?
            ''', (since_date,))
            avg_tokens = cursor.fetchone()[0] or 0
            
            # Trends data
            cursor.execute('''
                SELECT timestamp, response_time, confidence_score 
                FROM messages 
                WHERE role = 'assistant' AND timestamp >= ?
                ORDER BY timestamp
            ''', (since_date,))
            trends = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_messages': total_messages,
                'user_assistant_ratio': ratio,
                'avg_response_time': avg_response_time,
                'avg_confidence': avg_confidence,
                'mode_usage': modes,
                'avg_tokens': avg_tokens,
                'trends': trends
            }
            
        except sqlite3.Error as e:
            logger.error(f"Analytics query error: {e}")
            return {
                'total_messages': 0, 'user_assistant_ratio': '0:0',
                'avg_response_time': 0, 'avg_confidence': 0,
                'mode_usage': {}, 'avg_tokens': 0, 'trends': []
            }
        except Exception as e:
            logger.error(f"Unexpected analytics error: {e}")
            return {
                'total_messages': 0, 'user_assistant_ratio': '0:0',
                'avg_response_time': 0, 'avg_confidence': 0,
                'mode_usage': {}, 'avg_tokens': 0, 'trends': []
            }

    def search_messages(self, query: str, conversation_id: Optional[str] = None) -> List[ChatMessage]:
        """Search messages by keyword"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            sql = '''
                SELECT id, role, content, timestamp, response_time, confidence_score, 
                       sources, metadata, feedback, conversation_id
                FROM messages 
                WHERE content LIKE ?
            '''
            params = [f'%{query}%']
            
            if conversation_id:
                sql += ' AND conversation_id = ?'
                params.append(conversation_id)
            
            sql += ' ORDER BY timestamp DESC'
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            conn.close()
            
            messages = []
            for row in rows:
                sources = json.loads(row[6]) if row[6] else None
                metadata = json.loads(row[7]) if row[7] else {}
                messages.append(ChatMessage(
                    id=row[0],
                    role=row[1],
                    content=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    metadata=metadata,
                    confidence_score=row[5],
                    sources=sources,
                    response_time=row[4],
                    feedback=row[8]
                ))
            return messages
        except sqlite3.Error as e:
            logger.error(f"Search messages error: {e}")
            return []

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
    
    def process_document(self, file_content: bytes, filename: str, chunk_size: int, chunk_overlap: int, use_ocr: bool = False) -> Tuple[List[Document], DocumentMetadata]:
        """Process uploaded document with comprehensive error handling"""
        if not self.embeddings:
            raise ValueError("Embeddings not initialized")
            
        file_type = filename.lower().split('.')[-1] if '.' in filename else 'unknown'
        checksum = hashlib.md5(file_content).hexdigest()
        
        content = None
        if use_ocr and file_type == 'pdf' and OCR_AVAILABLE:
            try:
                images = convert_from_bytes(file_content)
                text_pages = []
                for image in images:
                    text = pytesseract.image_to_string(image)
                    text_pages.append(text)
                content = '\n\n'.join(text_pages)
            except Exception as e:
                logger.warning(f"OCR failed for {filename}: {e}")
                content = None
        
        temp_file = None
        try:
            if content is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
                    tmp_file.write(content.encode('utf-8'))
                    tmp_file.flush()
                    temp_file = tmp_file.name
                    loader = TextLoader(temp_file, encoding='utf-8')
            else:
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
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        try:
            chunks = text_splitter.split_documents(docs)
        except Exception as e:
            raise ValueError(f"Failed to split document {filename}: {e}")
        
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
# Streaming Callback
# -----------------------
class StreamingCallback(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.content = ""
        self.start_time = time.time()
        self.token_count = 0
    
    def on_llm_new_token(self, token: str, **kwargs):
        self.content += token
        self.token_count += 1
        self.placeholder.markdown(self.content)
    
    def get_metrics(self):
        elapsed = time.time() - self.start_time
        tps = self.token_count / elapsed if elapsed > 0 else 0
        return elapsed, tps

# -----------------------
# Chat Engine
# -----------------------
class ChatEngine:
    def __init__(self):
        self.db_manager = DatabaseManager(DB_PATH)
        self.doc_processor = DocumentProcessor()
        self.memory = ConversationBufferMemory(return_messages=True)
        self.search = DuckDuckGoSearchRun()
    
    def get_llm(self, model_name: str, temperature: float, base_url: str, streaming: bool = False, callbacks: List = None):
        """Create LLM instance with error handling"""
        try:
            if streaming and callbacks:
                return ChatOllama(
                    model=model_name,
                    base_url=base_url,
                    temperature=temperature,
                    streaming=True,
                    callbacks=callbacks
                )
            else:
                return OllamaLLM(
                    model=model_name,
                    base_url=base_url,
                    temperature=temperature,
                    timeout=30,
                    callbacks=callbacks if streaming else None
                )
        except Exception as e:
            logger.error(f"Failed to create LLM: {e}")
            raise
    
    def test_connection(self, model_name: str, base_url: str) -> bool:
        """Test Ollama connection with proper error handling"""
        try:
            llm = OllamaLLM(model=model_name, base_url=base_url, timeout=10)
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
        
        if len(response) > 50:
            score += 0.1
        
        if sources and len(sources) > 0:
            score += 0.2
        
        uncertainty_words = ['maybe', 'might', 'possibly', 'not sure', 'i think', 'perhaps']
        uncertainty_count = sum(1 for word in uncertainty_words if word in response.lower())
        score -= uncertainty_count * 0.1
        
        confidence_words = ['definitely', 'certainly', 'clearly', 'obviously']
        confidence_count = sum(1 for word in confidence_words if word in response.lower())
        score += confidence_count * 0.05
        
        return max(0.0, min(1.0, score))
    
    def generate_response(self, user_input: str, mode: str, system_prompt: str, 
                         model_config: Dict[str, Any], streaming_placeholder=None) -> ChatMessage:
        """Generate response based on mode with comprehensive error handling"""
        start_time = time.time()
        
        try:
            # Extract only required parameters for get_llm
            llm_config = {
                'model_name': model_config['model_name'],
                'temperature': model_config['temperature'],
                'base_url': model_config['base_url']
            }
            callbacks = [StreamingCallback(streaming_placeholder)] if streaming_placeholder else None
            llm = self.get_llm(**llm_config, streaming=bool(callbacks), callbacks=callbacks)
            
            if mode == "Chat":
                response, sources = self._generate_chat_response(user_input, system_prompt, llm, callbacks), None
            elif mode == "RAG":
                response, sources = self._generate_rag_response(user_input, system_prompt, llm, callbacks)
            elif mode == "WebSearch":
                response, sources = self._generate_websearch_response(user_input, system_prompt, llm, callbacks)
            else:
                response, sources = self._generate_chat_response(user_input, system_prompt, llm, callbacks), None
            
            if callbacks:
                response_time, tps = callbacks[0].get_metrics()
                model_config['tps'] = tps
            else:
                response_time = time.time() - start_time
            
            confidence_score = self.calculate_confidence_score(response, sources)
            
            return ChatMessage(
                id=str(uuid.uuid4()),
                role="assistant",
                content=response,
                timestamp=datetime.now(),
                metadata=model_config,
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
    
    def _generate_chat_response(self, user_input: str, system_prompt: str, llm, callbacks) -> str:
        """Generate chat response with memory and error handling"""
        try:
            if system_prompt and system_prompt.strip():
                full_prompt = f"System: {system_prompt}\n\nUser: {user_input}"
            else:
                full_prompt = user_input
                
            if callbacks:
                llm.invoke(full_prompt)
                return callbacks[0].content
            else:
                return llm.invoke(full_prompt)
            
        except Exception as e:
            logger.error(f"Chat response error: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _generate_rag_response(self, user_input: str, system_prompt: str, llm, callbacks) -> Tuple[str, List[str]]:
        """Generate RAG response using documents with error handling"""
        try:
            if not st.session_state.get('faiss_index') or not st.session_state.get('documents'):
                return "⚠️ No documents available. Please upload documents first.", []
            
            try:
                doc_scores = st.session_state.faiss_index.similarity_search_with_score(user_input, k=3)
                relevant_docs = [doc for doc, score in doc_scores]
                scores = [score for doc, score in doc_scores]
            except Exception as e:
                logger.error(f"Document retrieval error: {e}")
                return f"❌ Error retrieving documents: {str(e)}", []
            
            if not relevant_docs:
                return "❌ No relevant information found in documents.", []
            
            context_parts = []
            sources = []
            source_map = {}
            unique_sources = set()
            
            for i, (doc, score) in enumerate(zip(relevant_docs, scores), 1):
                if doc.page_content and doc.page_content.strip():
                    context_parts.append(f"[{i}] {doc.page_content.strip()}")
                    source_file = doc.metadata.get('source_file', 'Unknown')
                    if source_file not in unique_sources:
                        unique_sources.add(source_file)
                        source_map[i] = source_file
            
            if not context_parts:
                return "❌ No valid content found in retrieved documents.", []
            
            context = "\n\n".join(context_parts)
            
            prompt_parts = []
            if system_prompt and system_prompt.strip():
                prompt_parts.append(f"System: {system_prompt}")
            
            prompt_parts.extend([
                f"Context: {context}",
                f"Question: {user_input}",
                "Answer based on the context provided. Use inline citations like [1] when referencing specific parts of the context."
            ])
            
            prompt = "\n\n".join(prompt_parts)
            
            try:
                if callbacks:
                    llm.invoke(prompt)
                    response = callbacks[0].content
                else:
                    response = llm.invoke(prompt)
            except Exception as e:
                logger.error(f"LLM invocation error: {e}")
                return f"❌ Error generating response: {str(e)}", list(unique_sources)
            
            if source_map:
                sources_list = "\n\n**Sources:**\n" + "\n".join(f"[{i}] {source_map[i]}" for i in sorted(source_map))
                response += sources_list
            
            return response, list(unique_sources)
            
        except Exception as e:
            logger.error(f"RAG response error: {e}")
            return f"❌ Error in RAG processing: {str(e)}", []
    
    def _generate_websearch_response(self, user_input: str, system_prompt: str, llm, callbacks) -> Tuple[str, List[str]]:
        """Generate WebSearch response using DuckDuckGo"""
        try:
            search_results = self.search.run(user_input)
            sources = [search_results] if search_results else []
            
            context = search_results if search_results else "No search results found."
            
            prompt_parts = []
            if system_prompt and system_prompt.strip():
                prompt_parts.append(f"System: {system_prompt}")
            
            prompt_parts.extend([
                f"Search Context: {context}",
                f"Question: {user_input}",
                "Answer based on the search context provided. Cite sources where appropriate."
            ])
            
            prompt = "\n\n".join(prompt_parts)
            
            try:
                if callbacks:
                    llm.invoke(prompt)
                    response = callbacks[0].content
                else:
                    response = llm.invoke(prompt)
            except Exception as e:
                logger.error(f"LLM invocation error: {e}")
                return f"❌ Error generating response: {str(e)}", sources
            
            if sources:
                response += f"\n\n**Web Sources:** {search_results}"
            
            return response, sources
            
        except Exception as e:
            logger.error(f"WebSearch response error: {e}")
            return f"❌ Error in WebSearch processing: {str(e)}", []
    
    def generate_summary(self, messages: List[ChatMessage], llm) -> str:
        """Generate conversation summary"""
        try:
            conversation_text = "\n".join(f"{msg.role}: {msg.content}" for msg in messages)
            prompt = f"Summarize the following conversation concisely:\n\n{conversation_text}"
            return llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return "Summary generation failed."

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
            "system_prompt": "You are a helpful AI assistant. Provide accurate, concise, and helpful responses.",
            "pending_response": False,
            "search_query": "",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "use_ocr": False,
            "summarization_threshold": 10
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")) and not st.session_state.faiss_index:
            try:
                embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
                st.session_state.faiss_index = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
                st.session_state.documents = list(st.session_state.faiss_index.docstore._dict.values())
            except Exception as e:
                logger.error(f"Failed to load persistent FAISS: {e}")
        
        self.chat_engine.db_manager.save_conversation(st.session_state.conversation_id)
    
    def test_ollama_connection(self, base_url: str, model_name: str) -> bool:
        """Test Ollama connection with caching"""
        cache_key = f"connection_{base_url}_{model_name}"
        
        if cache_key in st.session_state:
            cached_time, cached_result = st.session_state[cache_key]
            if time.time() - cached_time < 30:
                return cached_result
        
        result = self.chat_engine.test_connection(model_name, base_url)
        
        st.session_state[cache_key] = (time.time(), result)
        
        return result
    
    def render_sidebar(self):
        """Render sidebar for multi-conversation management"""
        with st.sidebar:
            st.subheader("📝 Conversations")
            
            conversations = self.chat_engine.db_manager.get_conversations()
            for conv in conversations:
                if st.button(f"{conv['title']} ({conv['message_count']} msgs)", key=f"conv_{conv['id']}"):
                    self.switch_conversation(conv['id'])
            
            if st.button("➕ New Conversation"):
                new_id = str(uuid.uuid4())
                self.chat_engine.db_manager.save_conversation(new_id)
                self.switch_conversation(new_id)
            
            st.subheader("Current Conversation")
            title = st.text_input("Title", value=self.get_conversation_title(), key="conv_title")
            if st.button("Save Title"):
                self.chat_engine.db_manager.update_conversation(st.session_state.conversation_id, title=title)
                st.rerun()
    
    def get_conversation_title(self) -> str:
        """Get current conversation title"""
        convs = self.chat_engine.db_manager.get_conversations()
        for conv in convs:
            if conv['id'] == st.session_state.conversation_id:
                return conv['title']
        return f"Conversation {st.session_state.conversation_id[:8]}"
    
    def switch_conversation(self, conv_id: str):
        """Switch to another conversation"""
        st.session_state.conversation_id = conv_id
        st.session_state.messages = self.chat_engine.db_manager.load_messages(conv_id)
        st.session_state.pending_response = False
        st.rerun()
    
    def render_control_panel(self):
        """Render the main control panel"""
        with st.container():
            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns([1.5, 1.5, 1.5, 1.5, 1])
            
            with col1:
                mode = st.selectbox(
                    "💬 Mode",
                    ["Chat", "RAG", "WebSearch"],
                    help="Chat: Conversation mode | RAG: Document Q&A | WebSearch: Internet search",
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
            
            with col5:
                web_search = st.checkbox("🌐 Web Search", value=False, key="web_search_toggle")
            
            is_connected = self.test_ollama_connection(base_url, model_name)
            self.ui.render_status_indicator(is_connected, model_name)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            return {
                "mode": mode,
                "model_name": model_name,
                "temperature": temperature,
                "base_url": base_url,
                "is_connected": is_connected,
                "web_search": web_search
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
        
        with st.expander("⚙️ Preprocessing Options"):
            st.session_state.chunk_size = st.slider("Chunk Size", 500, 2000, st.session_state.chunk_size)
            st.session_state.chunk_overlap = st.slider("Chunk Overlap", 0, 500, st.session_state.chunk_overlap)
            if OCR_AVAILABLE:
                st.session_state.use_ocr = st.checkbox("Run OCR on PDFs", value=st.session_state.use_ocr)
            else:
                st.info("OCR not available. Install pdf2image and pytesseract for OCR support.")
        
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
                st.success(f"📄 {len(st.session_state.documents)} chunks loaded from {len(set(d.metadata['source_file'] for d in st.session_state.documents))} documents")
                if st.button("🗑️ Clear All", use_container_width=True, key="clear_docs"):
                    self.clear_documents()
                    st.rerun()
            else:
                st.info("No documents loaded")
        
        if uploaded_files:
            if st.button("📥 Process Documents", type="primary", use_container_width=True, key="process_docs"):
                self.process_documents(uploaded_files)
                st.rerun()
        
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
                
                if file.size > 10 * 1024 * 1024:
                    failed_files.append(f"{file.name} (file too large)")
                    continue
                
                chunks, metadata = self.chat_engine.doc_processor.process_document(
                    file.getvalue(), file.name, 
                    st.session_state.chunk_size, st.session_state.chunk_overlap,
                    st.session_state.use_ocr
                )
                
                if chunks:
                    all_chunks.extend(chunks)
                    processed_files += 1
                else:
                    failed_files.append(f"{file.name} (no content extracted)")
                
            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")
                failed_files.append(f"{file.name} ({str(e)})")
        
        if all_chunks:
            try:
                status_text.text("Creating search index...")
                embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
                if st.session_state.faiss_index:
                    st.session_state.faiss_index.add_documents(all_chunks)
                else:
                    st.session_state.faiss_index = FAISS.from_documents(all_chunks, embeddings)
                st.session_state.faiss_index.save_local(INDEX_DIR)
                st.session_state.documents.extend(all_chunks)
                
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
        if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
            os.remove(os.path.join(INDEX_DIR, "index.faiss"))
            os.remove(os.path.join(INDEX_DIR, "index.pkl"))
        st.toast("Documents cleared! 🗑️")
    
    def render_chat_interface(self, config: Dict[str, Any]):
        """Render the chat interface"""
        st.session_state.search_query = st.text_input("🔍 Search Chat History", value=st.session_state.search_query)
        
        if st.session_state.search_query:
            messages_to_show = self.chat_engine.db_manager.search_messages(st.session_state.search_query, st.session_state.conversation_id)
        else:
            messages_to_show = st.session_state.messages
        
        for message in messages_to_show:
            self.render_message(message)
        
        user_input = st.chat_input(
            "Type your message here...", 
            disabled=not config["is_connected"],
            key="chat_input"
        )
        
        if user_input and config["is_connected"]:
            user_message = ChatMessage(
                id=str(uuid.uuid4()),
                role="user",
                content=user_input,
                timestamp=datetime.now(),
                metadata={}
            )
            st.session_state.messages.append(user_message)
            self.chat_engine.db_manager.save_message(user_message, st.session_state.conversation_id)
            st.session_state.pending_response = True
            st.rerun()
        
        if st.session_state.pending_response:
            with st.chat_message("assistant"):
                self.ui.render_typing_indicator()
                placeholder = st.empty()
                with st.spinner("🤔 Thinking..."):
                    ai_message = self.chat_engine.generate_response(
                        user_input=st.session_state.messages[-1].content,
                        mode=config["mode"],
                        system_prompt=config["system_prompt"],
                        model_config={
                            "model_name": config["model_name"],
                            "temperature": config["temperature"],
                            "base_url": config["base_url"],
                            "mode": config["mode"]
                        },
                        streaming_placeholder=placeholder
                    )
                    st.session_state.messages.append(ai_message)
                    self.chat_engine.db_manager.save_message(ai_message, st.session_state.conversation_id)
                    
                    if len(st.session_state.messages) % st.session_state.summarization_threshold == 0:
                        llm = self.chat_engine.get_llm(config["model_name"], config["temperature"], config["base_url"])
                        summary = self.chat_engine.generate_summary(st.session_state.messages, llm)
                        self.chat_engine.db_manager.update_conversation(st.session_state.conversation_id, summary=summary)
            
            st.session_state.pending_response = False
            st.rerun()
    
    def render_message(self, message: ChatMessage):
        """Render a single message with enhanced UI"""
        with st.chat_message(message.role):
            st.markdown(message.content)
            
            if message.sources and message.role == "assistant" and message.metadata.get('mode') == "RAG":
                with st.expander("📑 Retrieved Contexts"):
                    doc_scores = st.session_state.faiss_index.similarity_search_with_score(message.content, k=3) if st.session_state.faiss_index else []
                    for doc, score in doc_scores:
                        st.write(f"**Similarity: {score:.2f}**")
                        st.code(doc.page_content)
                        st.caption(f"From: {doc.metadata.get('source_file', 'Unknown')}")
            
            if message.sources and message.role == "assistant" and message.metadata.get('mode') == "WebSearch":
                with st.expander("🌐 Web Sources"):
                    for source in message.sources:
                        st.write(source)
            
            if message.role == "assistant" and (message.confidence_score is not None or message.response_time is not None):
                col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                
                with col1:
                    if message.confidence_score is not None:
                        confidence_html = self.ui.render_confidence_badge(message.confidence_score)
                        st.markdown(confidence_html, unsafe_allow_html=True)
                
                with col2:
                    if message.response_time is not None:
                        st.caption(f"⏱️ {message.response_time:.1f}s")
                
                with col3:
                    tps = message.metadata.get('tps')
                    if tps:
                        st.caption(f"⚡ {tps:.1f} t/s")
                
                with col4:
                    if st.button("👍", key=f"like_{message.id}", help="Good response"):
                        self.chat_engine.db_manager.update_message_feedback(message.id, 1)
                        st.toast("Thanks for the feedback! 👍")
                
                with col5:
                    if st.button("👎", key=f"dislike_{message.id}", help="Poor response"):
                        self.chat_engine.db_manager.update_message_feedback(message.id, -1)
                        st.toast("Thanks for the feedback! We'll improve.")
    
    def render_analytics_tab(self):
        """Render analytics dashboard"""
        st.subheader("📊 Analytics Dashboard")
        
        analytics = self.chat_engine.db_manager.get_analytics_summary()
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            self.ui.render_metric_card(
                str(analytics['total_messages']), 
                "Total Messages", 
                "💬"
            )
        
        with col2:
            self.ui.render_metric_card(
                analytics['user_assistant_ratio'], 
                "User:AI Ratio", 
                "⚖️"
            )
        
        with col3:
            self.ui.render_metric_card(
                f"{analytics['avg_response_time']:.1f}s", 
                "Avg Response Time", 
                "⏱️"
            )
        
        with col4:
            self.ui.render_metric_card(
                f"{analytics['avg_confidence']:.2f}", 
                "Avg Confidence", 
                "🎯"
            )
        
        with col5:
            self.ui.render_metric_card(
                f"{analytics['avg_tokens']:.0f}", 
                "Avg Tokens/Response", 
                "📝"
            )
        
        with col6:
            chat_usage = analytics['mode_usage'].get('Chat', 0)
            rag_usage = analytics['mode_usage'].get('RAG', 0)
            web_usage = analytics['mode_usage'].get('WebSearch', 0)
            self.ui.render_metric_card(
                f"{chat_usage}/{rag_usage}/{web_usage}", 
                "Chat/RAG/Web", 
                "📊"
            )
        
        if analytics['trends']:
            df = pd.DataFrame(analytics['trends'], columns=['timestamp', 'response_time', 'confidence_score'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            st.subheader("📈 Trends")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_rt = px.line(df, x='timestamp', y='response_time', title='Response Time Trend')
                st.plotly_chart(fig_rt, use_container_width=True)
            
            with col2:
                fig_conf = px.line(df, x='timestamp', y='confidence_score', title='Confidence Trend')
                st.plotly_chart(fig_conf, use_container_width=True)
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
            st.session_state.summarization_threshold = st.number_input("Summarize every X messages", min_value=5, max_value=50, value=st.session_state.summarization_threshold)
            
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
                self.chat_engine.db_manager.update_conversation(st.session_state.conversation_id, summary=None)
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
        keys_to_keep = ['system_prompt']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        
        self.init_session_state()
        self.chat_engine.memory.clear()
        
        st.toast("Application reset! 🔄")
    
    def run(self):
        """Run the main application"""
        st.set_page_config(
            page_title="Chat With Gemma",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.ui.apply_custom_css()
        
        self.render_sidebar()
        
        self.ui.render_header()
        
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📚 Documents", "📊 Analytics", "⚙️ Settings"])
        
        with tab1:
            config = self.render_control_panel()
            
            system_prompt = self.render_system_prompt_section()
            config["system_prompt"] = system_prompt
            
            col1, col2, col3 = st.columns([1, 1, 4])
            
            with col1:
                if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_main"):
                    st.session_state.messages = []
                    st.session_state.conversation_id = str(uuid.uuid4())
                    self.chat_engine.db_manager.save_conversation(st.session_state.conversation_id)
                    self.chat_engine.memory.clear()
                    st.rerun()
            
            with col2:
                if st.button("💾 Save Chat", use_container_width=True, key="save_chat_main"):
                    st.toast("Chat saved! 💾")
            
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
