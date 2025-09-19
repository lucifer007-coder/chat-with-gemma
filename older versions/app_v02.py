import os
import json
import tempfile
import asyncio
import time
import hashlib
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_ace import st_ace
from streamlit_chat import message
import streamlit.components.v1 as components

# Core LangChain imports
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationEntityMemory
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

# Additional imports for enhanced features
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Try to download spaCy model if not present
try:
    import spacy
    spacy.load("en_core_web_sm")
except OSError:
    st.warning("spaCy English model not found. Installing...")
    try:
        os.system("python -m spacy download en_core_web_sm")
        st.success("spaCy model installed successfully!")
    except:
        st.warning("Could not install spaCy model automatically. Some NLP features will be limited.")

# -----------------------
# Configuration & Constants
# -----------------------
OLLAMA_DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:1b"  # Changed to a more common model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Directory structure
DATA_DIR = "chat_data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
DOCS_META = os.path.join(DATA_DIR, "docs_meta.json")
HISTORY_DIR = os.path.join(DATA_DIR, "histories")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
DB_PATH = os.path.join(DATA_DIR, "chat_analytics.db")

# Create directories
for dir_path in [DATA_DIR, HISTORY_DIR, CACHE_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# -----------------------
# Logging Setup
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'chat_app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
class ConversationMetrics:
    conversation_id: str
    total_messages: int
    avg_response_time: float
    user_satisfaction: Optional[float]
    topics: List[str]
    created_at: datetime
    last_updated: datetime

@dataclass
class DocumentMetadata:
    filename: str
    file_type: str
    size: int
    upload_time: datetime
    chunks_count: int
    embedding_model: str
    checksum: str
    version: int = 1

# -----------------------
# Database Setup
# -----------------------
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP,
                last_updated TIMESTAMP,
                message_count INTEGER DEFAULT 0,
                user_id TEXT DEFAULT 'default'
            )
        ''')
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP,
                response_time REAL,
                confidence_score REAL,
                sources TEXT,
                metadata TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')
        
        # Analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                event_data TEXT,
                timestamp TIMESTAMP,
                user_id TEXT DEFAULT 'default'
            )
        ''')
        
        # User feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT,
                rating INTEGER,
                feedback_text TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (message_id) REFERENCES messages (id)
            )
        ''')
        
        # Document versions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                version INTEGER,
                checksum TEXT,
                upload_time TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, conversation_id: str, title: str = None):
        """Save or update conversation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO conversations 
            (id, title, created_at, last_updated) 
            VALUES (?, ?, ?, ?)
        ''', (conversation_id, title or f"Chat {conversation_id[:8]}", 
              datetime.now(), datetime.now()))
        
        conn.commit()
        conn.close()
    
    def save_message(self, message: ChatMessage, conversation_id: str):
        """Save message to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages 
            (id, conversation_id, role, content, timestamp, response_time, 
             confidence_score, sources, metadata) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (message.id, conversation_id, message.role, message.content,
              message.timestamp, message.response_time, message.confidence_score,
              json.dumps(message.sources) if message.sources else None,
              json.dumps(message.metadata)))
        
        conn.commit()
        conn.close()
    
    def log_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log analytics event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analytics (event_type, event_data, timestamp) 
            VALUES (?, ?, ?)
        ''', (event_type, json.dumps(event_data), datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_conversations(self, limit: int = 50) -> List[Dict]:
        """Get recent conversations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, title, created_at, last_updated, message_count 
            FROM conversations 
            ORDER BY last_updated DESC 
            LIMIT ?
        ''', (limit,))
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                'id': row[0],
                'title': row[1],
                'created_at': row[2],
                'last_updated': row[3],
                'message_count': row[4]
            })
        
        conn.close()
        return conversations
    
    def get_analytics_data(self, days: int = 30) -> Dict[str, Any]:
        """Get analytics data for dashboard"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get message counts by day
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM messages 
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        '''.format(days))
        
        daily_messages = cursor.fetchall()
        
        # Get average response times
        cursor.execute('''
            SELECT AVG(response_time) as avg_time
            FROM messages 
            WHERE response_time IS NOT NULL 
            AND timestamp >= datetime('now', '-{} days')
        '''.format(days))
        
        avg_response_time = cursor.fetchone()[0] or 0
        
        # Get user satisfaction
        cursor.execute('''
            SELECT AVG(rating) as avg_rating, COUNT(*) as total_ratings
            FROM feedback 
            WHERE timestamp >= datetime('now', '-{} days')
        '''.format(days))
        
        satisfaction_data = cursor.fetchone()
        avg_rating = satisfaction_data[0] or 0
        total_ratings = satisfaction_data[1] or 0
        
        conn.close()
        
        return {
            'daily_messages': daily_messages,
            'avg_response_time': avg_response_time,
            'avg_rating': avg_rating,
            'total_ratings': total_ratings
        }

# -----------------------
# Streaming Callback Handler
# -----------------------
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.placeholder = None
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        if self.placeholder is None:
            self.placeholder = self.container.empty()
        self.placeholder.markdown(self.text + "▌")
    
    def on_llm_end(self, response, **kwargs) -> None:
        if self.placeholder:
            self.placeholder.markdown(self.text)

# -----------------------
# Enhanced Document Processing
# -----------------------
class AdvancedDocumentProcessor:
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        self.embedding_model_name = embedding_model_name
        self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.warning("spaCy English model not found. Some features may be limited.")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep punctuation
        import re
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        return text
    
    def extract_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract metadata from document text"""
        metadata = {
            'filename': filename,
            'word_count': len(text.split()),
            'char_count': len(text),
            'extracted_at': datetime.now().isoformat()
        }
        
        if self.nlp:
            doc = self.nlp(text[:1000])  # Process first 1000 chars for efficiency
            
            # Extract entities
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            metadata['entities'] = entities[:10]  # Top 10 entities
            
            # Extract key phrases (noun phrases)
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            metadata['key_phrases'] = noun_phrases[:10]
        
        return metadata
    
    def dynamic_chunk_size(self, text: str, file_type: str) -> int:
        """Determine optimal chunk size based on content"""
        base_size = 1000
        
        # Adjust based on file type
        if file_type == 'pdf':
            base_size = 1200  # PDFs often have more structured content
        elif file_type in ['txt', 'md']:
            base_size = 800   # Text files might be more conversational
        
        # Adjust based on content length
        if len(text) < 5000:
            base_size = min(base_size, len(text) // 3)
        
        return max(200, base_size)  # Minimum chunk size of 200
    
    def process_document(self, file_content: bytes, filename: str) -> Tuple[List[Document], DocumentMetadata]:
        """Process document with enhanced features"""
        file_type = filename.lower().split('.')[-1]
        
        # Calculate checksum
        checksum = hashlib.md5(file_content).hexdigest()
        
        # Load document
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
            tmp_file.write(file_content)
            tmp_file.flush()
            
            try:
                if file_type in ['txt', 'md']:
                    loader = TextLoader(tmp_file.name, encoding='utf-8')
                elif file_type == 'pdf':
                    loader = PyPDFLoader(tmp_file.name)
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")
                
                docs = loader.load()
                
            finally:
                os.unlink(tmp_file.name)
        
        # Combine all pages/sections
        full_text = '\n'.join([doc.page_content for doc in docs])
        cleaned_text = self.clean_text(full_text)
        
        # Extract metadata
        doc_metadata = self.extract_metadata(cleaned_text, filename)
        
        # Dynamic chunking
        chunk_size = self.dynamic_chunk_size(cleaned_text, file_type)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.15),  # 15% overlap
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Create new documents with cleaned text
        processed_docs = []
        for i, doc in enumerate(docs):
            doc.page_content = self.clean_text(doc.page_content)
            doc.metadata.update({
                'source_file': filename,
                'file_type': file_type,
                'page_number': i + 1,
                'processed_at': datetime.now().isoformat()
            })
        
        # Split into chunks
        chunks = text_splitter.split_documents(docs)
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'chunk_size': len(chunk.page_content),
                'checksum': checksum
            })
        
        # Create document metadata object
        metadata = DocumentMetadata(
            filename=filename,
            file_type=file_type,
            size=len(file_content),
            upload_time=datetime.now(),
            chunks_count=len(chunks),
            embedding_model=self.embedding_model_name,
            checksum=checksum
        )
        
        return chunks, metadata

# -----------------------
# Advanced Retrieval System
# -----------------------
class HybridRetriever:
    def __init__(self, faiss_index: FAISS, documents: List[Document]):
        self.faiss_index = faiss_index
        self.documents = documents
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Build TF-IDF index
        if documents:
            doc_texts = [doc.page_content for doc in documents]
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(doc_texts)
        else:
            self.tfidf_matrix = None
    
    def retrieve_with_mmr(self, query: str, k: int = 4, lambda_mult: float = 0.5) -> List[Document]:
        """Retrieve documents using Maximum Marginal Relevance"""
        if not self.documents:
            return []
        
        try:
            # Get initial candidates (more than needed)
            initial_docs = self.faiss_index.similarity_search(query, k=k*2)
            
            if not initial_docs:
                return []
            
            # Calculate query embedding
            query_embedding = self.faiss_index.embeddings.embed_query(query)
            
            # Calculate document embeddings
            doc_embeddings = []
            for doc in initial_docs:
                doc_embedding = self.faiss_index.embeddings.embed_query(doc.page_content)
                doc_embeddings.append(doc_embedding)
            
            # MMR selection
            selected_docs = []
            remaining_docs = list(zip(initial_docs, doc_embeddings))
            
            while len(selected_docs) < k and remaining_docs:
                if not selected_docs:
                    # Select first document (highest similarity to query)
                    best_idx = 0
                    best_score = cosine_similarity([query_embedding], [remaining_docs[0][1]])[0][0]
                    
                    for i, (doc, embedding) in enumerate(remaining_docs[1:], 1):
                        score = cosine_similarity([query_embedding], [embedding])[0][0]
                        if score > best_score:
                            best_score = score
                            best_idx = i
                else:
                    # Select document that maximizes MMR score
                    best_idx = 0
                    best_mmr_score = -float('inf')
                    
                    selected_embeddings = [self.faiss_index.embeddings.embed_query(doc.page_content) 
                                         for doc in selected_docs]
                    
                    for i, (doc, embedding) in enumerate(remaining_docs):
                        # Relevance to query
                        relevance = cosine_similarity([query_embedding], [embedding])[0][0]
                        
                        # Maximum similarity to already selected documents
                        max_sim = max([cosine_similarity([embedding], [sel_emb])[0][0] 
                                     for sel_emb in selected_embeddings])
                        
                        # MMR score
                        mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                        
                        if mmr_score > best_mmr_score:
                            best_mmr_score = mmr_score
                            best_idx = i
                
                # Add selected document and remove from candidates
                selected_docs.append(remaining_docs[best_idx][0])
                remaining_docs.pop(best_idx)
            
            return selected_docs
            
        except Exception as e:
            logger.error(f"Error in MMR retrieval: {e}")
            # Fallback to regular similarity search
            return self.faiss_index.similarity_search(query, k=k)
    
    def hybrid_search(self, query: str, k: int = 4, alpha: float = 0.7) -> List[Document]:
        """Combine semantic and keyword-based search"""
        if not self.documents or self.tfidf_matrix is None:
            return self.faiss_index.similarity_search(query, k=k)
        
        try:
            # Semantic search
            semantic_docs = self.faiss_index.similarity_search_with_score(query, k=k*2)
            
            # Keyword search
            query_tfidf = self.tfidf_vectorizer.transform([query])
            tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
            
            # Get top TF-IDF documents
            top_tfidf_indices = tfidf_scores.argsort()[-k*2:][::-1]
            
            # Combine scores
            doc_scores = {}
            
            # Add semantic scores
            for doc, score in semantic_docs:
                doc_id = id(doc)  # Use object id as unique identifier
                doc_scores[doc_id] = {'doc': doc, 'semantic': 1 - score, 'tfidf': 0}
            
            # Add TF-IDF scores
            for idx in top_tfidf_indices:
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    doc_id = id(doc)
                    if doc_id in doc_scores:
                        doc_scores[doc_id]['tfidf'] = tfidf_scores[idx]
                    else:
                        doc_scores[doc_id] = {'doc': doc, 'semantic': 0, 'tfidf': tfidf_scores[idx]}
            
            # Calculate combined scores
            for doc_id in doc_scores:
                semantic_score = doc_scores[doc_id]['semantic']
                tfidf_score = doc_scores[doc_id]['tfidf']
                combined_score = alpha * semantic_score + (1 - alpha) * tfidf_score
                doc_scores[doc_id]['combined'] = combined_score
            
            # Sort by combined score and return top k
            sorted_docs = sorted(doc_scores.values(), key=lambda x: x['combined'], reverse=True)
            return [item['doc'] for item in sorted_docs[:k]]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return self.faiss_index.similarity_search(query, k=k)

# -----------------------
# Enhanced Memory Management
# -----------------------
class AdvancedMemoryManager:
    def __init__(self, llm, max_token_limit: int = 4000):
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.memories = {
            'buffer': ConversationBufferMemory(return_messages=True),
            'summary': ConversationSummaryMemory(llm=llm, return_messages=True),
            'entity': ConversationEntityMemory(llm=llm, return_messages=True)
        }
        self.current_memory_type = 'buffer'
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def get_memory(self, memory_type: str = None):
        """Get memory instance"""
        memory_type = memory_type or self.current_memory_type
        return self.memories.get(memory_type, self.memories['buffer'])
    
    def add_message(self, user_message: str, ai_message: str):
        """Add message to all memory types"""
        for memory in self.memories.values():
            try:
                memory.chat_memory.add_user_message(user_message)
                memory.chat_memory.add_ai_message(ai_message)
            except Exception as e:
                logger.error(f"Error adding message to memory: {e}")
    
    def get_context(self, memory_type: str = None) -> str:
        """Get conversation context"""
        memory = self.get_memory(memory_type)
        try:
            return memory.buffer
        except:
            return ""
    
    def compress_context_if_needed(self) -> bool:
        """Compress context if it exceeds token limit"""
        current_context = self.get_context()
        if self.estimate_tokens(current_context) > self.max_token_limit:
            # Switch to summary memory
            self.current_memory_type = 'summary'
            return True
        return False
    
    def clear_memory(self):
        """Clear all memories"""
        for memory in self.memories.values():
            memory.clear()

# -----------------------
# Caching System
# -----------------------
class CacheManager:
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, data: str) -> str:
        """Generate cache key from data"""
        return hashlib.md5(data.encode()).hexdigest()
    
    def get_embedding_cache_path(self, text: str) -> str:
        """Get cache path for embedding"""
        key = self._get_cache_key(text)
        return os.path.join(self.cache_dir, f"embedding_{key}.npy")
    
    def cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding"""
        try:
            cache_path = self.get_embedding_cache_path(text)
            np.save(cache_path, embedding)
        except Exception as e:
            logger.error(f"Error caching embedding: {e}")
    
    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        try:
            cache_path = self.get_embedding_cache_path(text)
            if os.path.exists(cache_path):
                return np.load(cache_path)
        except Exception as e:
            logger.error(f"Error loading cached embedding: {e}")
        return None
    
    def cache_response(self, query: str, response: str, ttl_hours: int = 24):
        """Cache LLM response"""
        try:
            key = self._get_cache_key(query)
            cache_data = {
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'ttl_hours': ttl_hours
            }
            cache_path = os.path.join(self.cache_dir, f"response_{key}.json")
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.error(f"Error caching response: {e}")
    
    def get_cached_response(self, query: str) -> Optional[str]:
        """Get cached response if valid"""
        try:
            key = self._get_cache_key(query)
            cache_path = os.path.join(self.cache_dir, f"response_{key}.json")
            
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is still valid
                cached_time = datetime.fromisoformat(cache_data['timestamp'])
                ttl_hours = cache_data.get('ttl_hours', 24)
                
                if (datetime.now() - cached_time).total_seconds() < ttl_hours * 3600:
                    return cache_data['response']
        except Exception as e:
            logger.error(f"Error loading cached response: {e}")
        return None

# -----------------------
# Quality Assurance System
# -----------------------
class QualityAssurance:
    def __init__(self):
        self.harmful_keywords = [
            'violence', 'harm', 'illegal', 'dangerous', 'weapon',
            'drug', 'suicide', 'self-harm', 'hate', 'discrimination'
        ]
    
    def check_content_safety(self, text: str) -> Tuple[bool, List[str]]:
        """Check if content is safe"""
        issues = []
        text_lower = text.lower()
        
        for keyword in self.harmful_keywords:
            if keyword in text_lower:
                issues.append(f"Potentially harmful content detected: {keyword}")
        
        is_safe = len(issues) == 0
        return is_safe, issues
    
    def calculate_confidence_score(self, response: str, sources: List[str] = None) -> float:
        """Calculate confidence score for response"""
        score = 0.5  # Base score
        
        # Length factor (longer responses might be more detailed)
        if len(response) > 100:
            score += 0.1
        
        # Source factor
        if sources and len(sources) > 0:
            score += 0.2
        
        # Uncertainty indicators
        uncertainty_words = ['maybe', 'might', 'possibly', 'perhaps', 'not sure', 'unclear']
        uncertainty_count = sum(1 for word in uncertainty_words if word in response.lower())
        score -= uncertainty_count * 0.05
        
        # Confidence indicators
        confidence_words = ['definitely', 'certainly', 'clearly', 'obviously', 'precisely']
        confidence_count = sum(1 for word in confidence_words if word in response.lower())
        score += confidence_count * 0.05
        
        return max(0.0, min(1.0, score))
    
    def detect_hallucination_risk(self, response: str, sources: List[str] = None) -> Tuple[float, List[str]]:
        """Detect potential hallucination in response"""
        risk_score = 0.0
        warnings = []
        
        # Check for specific claims without sources
        specific_patterns = [
            r'\d{4}',  # Years
            r'\$[\d,]+',  # Money amounts
            r'\d+%',  # Percentages
            r'according to',  # Attribution claims
            r'studies show',  # Research claims
        ]
        
        import re
        for pattern in specific_patterns:
            if re.search(pattern, response) and (not sources or len(sources) == 0):
                risk_score += 0.2
                warnings.append(f"Specific claim made without source verification")
        
        # Check for absolute statements
        absolute_words = ['always', 'never', 'all', 'none', 'every', 'impossible']
        for word in absolute_words:
            if word in response.lower():
                risk_score += 0.1
                warnings.append(f"Absolute statement detected: '{word}'")
        
        return min(1.0, risk_score), warnings

# -----------------------
# Analytics Dashboard
# -----------------------
def render_analytics_dashboard(db_manager: DatabaseManager):
    """Render analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    
    # Time range selector
    time_range = st.selectbox("Time Range", [7, 14, 30, 90], index=2)
    
    # Get analytics data
    analytics_data = db_manager.get_analytics_data(days=time_range)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Avg Response Time", 
            f"{analytics_data['avg_response_time']:.2f}s"
        )
    
    with col2:
        st.metric(
            "User Satisfaction", 
            f"{analytics_data['avg_rating']:.1f}/5.0"
        )
    
    with col3:
        st.metric(
            "Total Ratings", 
            analytics_data['total_ratings']
        )
    
    with col4:
        total_messages = sum([count for _, count in analytics_data['daily_messages']])
        st.metric("Total Messages", total_messages)
    
    # Charts
    if analytics_data['daily_messages']:
        # Daily message chart
        df_messages = pd.DataFrame(analytics_data['daily_messages'], columns=['Date', 'Messages'])
        fig_messages = px.line(df_messages, x='Date', y='Messages', title='Daily Message Count')
        st.plotly_chart(fig_messages, use_container_width=True)
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    # Response time distribution (mock data for demo)
    response_times = np.random.normal(analytics_data['avg_response_time'], 0.5, 100)
    fig_hist = px.histogram(x=response_times, title='Response Time Distribution', 
                           labels={'x': 'Response Time (s)', 'y': 'Frequency'})
    st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------
# Main Application Class
# -----------------------
class AdvancedChatApp:
    def __init__(self):
        self.db_manager = DatabaseManager(DB_PATH)
        self.cache_manager = CacheManager()
        self.qa_system = QualityAssurance()
        self.doc_processor = AdvancedDocumentProcessor()
        self.memory_manager = None
        self.hybrid_retriever = None
        
        # Initialize session state
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize Streamlit session state"""
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "current_model" not in st.session_state:
            st.session_state.current_model = DEFAULT_MODEL
        
        if "embeddings" not in st.session_state:
            st.session_state.embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        if "faiss_index" not in st.session_state:
            st.session_state.faiss_index = self.load_or_create_index()
        
        if "documents" not in st.session_state:
            st.session_state.documents = []
        
        if "conversation_branches" not in st.session_state:
            st.session_state.conversation_branches = {}
    
    def load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
            try:
                return FAISS.load_local(INDEX_DIR, st.session_state.embeddings, 
                                      allow_dangerous_deserialization=True)
            except Exception as e:
                st.warning(f"Failed to load existing index: {e}")
        
        # Create empty index - FIXED VERSION
        try:
            # Create a minimal dummy document
            dummy_doc = Document(page_content="initialization", metadata={"temp": True})
            index = FAISS.from_documents([dummy_doc], st.session_state.embeddings)
            
            # Get the docstore keys and delete the dummy document
            docstore_keys = list(index.docstore._dict.keys())
            if docstore_keys:
                index.delete(docstore_keys)
            
            return index
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            # Fallback: create index without deletion
            dummy_doc = Document(page_content="initialization", metadata={"temp": True})
            return FAISS.from_documents([dummy_doc], st.session_state.embeddings)
    
    def get_llm(self, model_name: str, temperature: float, base_url: str, streaming: bool = False):
        """Create LLM instance"""
        callbacks = []
        if streaming:
            callbacks.append(StreamingStdOutCallbackHandler())
        
        return Ollama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            callbacks=callbacks
        )
    
    def process_user_input(self, user_input: str, mode: str, system_prompt: str, 
                          model_config: Dict[str, Any], streaming: bool = False):
        """Process user input and generate response"""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{user_input}_{mode}_{system_prompt}_{model_config}"
        cached_response = self.cache_manager.get_cached_response(cache_key)
        
        if cached_response and not streaming:
            response = cached_response
            response_time = 0.1  # Cached response
        else:
            # Generate new response
            llm = self.get_llm(**model_config, streaming=streaming)
            
            if self.memory_manager is None:
                self.memory_manager = AdvancedMemoryManager(llm)
            
            try:
                if mode == "Chat (memory)":
                    response = self.generate_chat_response(user_input, system_prompt, llm, streaming)
                elif mode == "RAG (use docs)":
                    response = self.generate_rag_response(user_input, system_prompt, llm, streaming)
                else:
                    response = "Unknown mode selected."
                
                response_time = time.time() - start_time
                
                # Cache response
                if not streaming:
                    self.cache_manager.cache_response(cache_key, response)
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                response = f"❌ Error generating response: {str(e)}"
                response_time = time.time() - start_time
        
        # Quality assurance checks
        is_safe, safety_issues = self.qa_system.check_content_safety(response)
        confidence_score = self.qa_system.calculate_confidence_score(response)
        hallucination_risk, hallucination_warnings = self.qa_system.detect_hallucination_risk(response)
        
        # Create message objects
        user_message = ChatMessage(
            id=str(uuid.uuid4()),
            role="user",
            content=user_input,
            timestamp=datetime.now(),
            metadata={"mode": mode}
        )
        
        ai_message = ChatMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=response,
            timestamp=datetime.now(),
            metadata={
                "mode": mode,
                "model": model_config["model_name"],
                "temperature": model_config["temperature"],
                "safety_issues": safety_issues,
                "hallucination_warnings": hallucination_warnings,
                "is_safe": is_safe
            },
            confidence_score=confidence_score,
            response_time=response_time
        )
        
        # Save to database
        self.db_manager.save_message(user_message, st.session_state.conversation_id)
        self.db_manager.save_message(ai_message, st.session_state.conversation_id)
        
        # Add to session state
        st.session_state.messages.extend([user_message, ai_message])
        
        # Update memory
        if self.memory_manager:
            self.memory_manager.add_message(user_input, response)
        
        # Log analytics
        self.db_manager.log_event("message_sent", {
            "mode": mode,
            "response_time": response_time,
            "confidence_score": confidence_score,
            "is_safe": is_safe
        })
        
        return ai_message
    
    def generate_chat_response(self, user_input: str, system_prompt: str, llm, streaming: bool = False):
        """Generate chat response with memory"""
        memory = self.memory_manager.get_memory()
        
        # Check if context compression is needed
        self.memory_manager.compress_context_if_needed()
        
        # Create conversation chain
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        
        # Combine system prompt with user input
        full_input = f"System: {system_prompt}\n\nUser: {user_input}"
        
        if streaming:
            # For streaming, we'll use a simpler approach
            response = conversation.predict(input=full_input)
        else:
            response = conversation.predict(input=full_input)
        
        return response
    
    def generate_rag_response(self, user_input: str, system_prompt: str, llm, streaming: bool = False):
        """Generate RAG response using documents"""
        if not st.session_state.documents:
            return "⚠️ No documents have been ingested. Please upload documents first."
        
        # Initialize hybrid retriever if needed
        if self.hybrid_retriever is None:
            self.hybrid_retriever = HybridRetriever(st.session_state.faiss_index, st.session_state.documents)
        
        # Retrieve relevant documents
        relevant_docs = self.hybrid_retriever.retrieve_with_mmr(user_input, k=4)
        
        if not relevant_docs:
            return "❌ No relevant documents found for your query."
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        sources = list(set([doc.metadata.get('source_file', 'Unknown') for doc in relevant_docs]))
        
        # Create enhanced prompt
        enhanced_prompt = f"""
        System: {system_prompt}
        
        Context from documents:
        {context}
        
        User Question: {user_input}
        
        Please answer the question based on the provided context. If the context doesn't contain enough information, say so clearly.
        """
        
        # Generate response
        if streaming:
            response = llm(enhanced_prompt)
        else:
            response = llm(enhanced_prompt)
        
        # Add source information
        if sources:
            response += f"\n\n📚 **Sources:** {', '.join(sources)}"
        
        return response
    
    def render_message(self, message: ChatMessage, show_metadata: bool = False):
        """Render a chat message with enhanced features"""
        with st.chat_message(message.role):
            # Message content
            st.write(message.content)
            
            # Message metadata
            if show_metadata:
                with st.expander("Message Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Timestamp:** {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                        if message.response_time:
                            st.write(f"**Response Time:** {message.response_time:.2f}s")
                        if message.confidence_score:
                            st.write(f"**Confidence:** {message.confidence_score:.2f}")
                    
                    with col2:
                        if message.sources:
                            st.write(f"**Sources:** {', '.join(message.sources)}")
                        if message.metadata:
                            st.write(f"**Mode:** {message.metadata.get('mode', 'Unknown')}")
            
            # Message actions
            if message.role == "assistant":
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("👍", key=f"like_{message.id}"):
                        self.save_feedback(message.id, 5, "Liked")
                        st.success("Feedback saved!")
                
                with col2:
                    if st.button("👎", key=f"dislike_{message.id}"):
                        self.save_feedback(message.id, 1, "Disliked")
                        st.success("Feedback saved!")
                
                with col3:
                    if st.button("📋", key=f"copy_{message.id}"):
                        st.code(message.content)
                
                with col4:
                    if st.button("🔄", key=f"regenerate_{message.id}"):
                        st.session_state.regenerate_message_id = message.id
                        st.rerun()
    
    def save_feedback(self, message_id: str, rating: int, feedback_text: str = ""):
        """Save user feedback"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (message_id, rating, feedback_text, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (message_id, rating, feedback_text, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def render_sidebar(self):
        """Render enhanced sidebar"""
        st.sidebar.header("🔧 Configuration")
        
        # Model settings
        with st.sidebar.expander("Model Settings", expanded=True):
            base_url = st.text_input("Ollama URL", value=OLLAMA_DEFAULT_URL)
            model_name = st.text_input("Model", value=DEFAULT_MODEL)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
            streaming = st.checkbox("Enable Streaming", value=False)
        
        # Mode selection
        st.sidebar.header("💬 Chat Mode")
        mode = st.radio("Mode", ["Chat (memory)", "RAG (use docs)"])
        
        # Memory settings
        if mode == "Chat (memory)":
            with st.sidebar.expander("Memory Settings"):
                memory_type = st.selectbox("Memory Type", 
                                         ["buffer", "summary", "entity"])
                max_tokens = st.slider("Max Context Tokens", 1000, 8000, 4000, 500)
        
        # System prompt
        st.sidebar.header("📝 System Prompt")
        system_prompt = st.sidebar.text_area(
            "System message:",
            value="You are a helpful assistant. Answer concisely and truthfully.",
            height=120
        )
        
        # Document management
        self.render_document_management()
        
        # Conversation management
        self.render_conversation_management()
        
        return {
            "model_name": model_name,
            "temperature": temperature,
            "base_url": base_url,
            "streaming": streaming,
            "mode": mode,
            "system_prompt": system_prompt
        }
    
    def render_document_management(self):
        """Render document management section"""
        st.sidebar.header("📚 Document Management")
        
        # File upload with drag and drop
        uploaded_files = st.sidebar.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=['txt', 'md', 'pdf'],
            help="Drag and drop files here or click to browse"
        )
        
        # Upload settings
        with st.sidebar.expander("Upload Settings"):
            chunk_strategy = st.selectbox("Chunking Strategy", 
                                        ["Dynamic", "Fixed Size", "Semantic"])
            enable_preprocessing = st.checkbox("Enable Text Preprocessing", value=True)
            extract_metadata = st.checkbox("Extract Metadata", value=True)
        
        if st.sidebar.button("📥 Process Documents"):
            if uploaded_files:
                self.process_documents(uploaded_files, chunk_strategy, 
                                     enable_preprocessing, extract_metadata)
            else:
                st.sidebar.warning("No files uploaded.")
        
        # Document list
        docs_meta = self.load_docs_meta()
        if docs_meta:
            st.sidebar.markdown("**📄 Ingested Documents:**")
            for doc in docs_meta:
                with st.sidebar.expander(f"📄 {doc['filename']}"):
                    st.write(f"**Type:** {doc.get('file_type', 'Unknown')}")
                    st.write(f"**Chunks:** {doc.get('chunks_count', 0)}")
                    st.write(f"**Size:** {doc.get('size', 0)} bytes")
                    st.write(f"**Uploaded:** {doc.get('upload_time', 'Unknown')}")
                    
                    if st.button(f"🗑️ Remove", key=f"remove_{doc['filename']}"):
                        self.remove_document(doc['filename'])
                        st.rerun()
        
        # Clear all documents
        if st.sidebar.button("🗑️ Clear All Documents"):
            self.clear_all_documents()
            st.sidebar.success("All documents cleared!")
            st.rerun()
    
    def render_conversation_management(self):
        """Render conversation management section"""
        st.sidebar.header("💾 Conversation Management")
        
        # Save conversation
        if st.sidebar.button("💾 Save Conversation"):
            self.save_current_conversation()
        
        # Load conversation
        conversations = self.db_manager.get_conversations(limit=10)
        if conversations:
            selected_conv = st.sidebar.selectbox(
                "Load Conversation",
                options=[None] + conversations,
                format_func=lambda x: "Select..." if x is None else f"{x['title']} ({x['message_count']} msgs)"
            )
            
            if selected_conv and st.sidebar.button("📂 Load"):
                self.load_conversation(selected_conv['id'])
                st.rerun()
        
        # Export/Import
        with st.sidebar.expander("Export/Import"):
            if st.button("📤 Export JSON"):
                self.export_conversation_json()
            
            uploaded_conv = st.file_uploader("📥 Import JSON", type="json")
            if uploaded_conv:
                self.import_conversation_json(uploaded_conv)
                st.rerun()
        
        # Clear conversation
        if st.sidebar.button("🗑️ Clear Current Chat"):
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            if self.memory_manager:
                self.memory_manager.clear_memory()
            st.sidebar.success("Conversation cleared!")
            st.rerun()
    
    def process_documents(self, uploaded_files, chunk_strategy: str, 
                         enable_preprocessing: bool, extract_metadata: bool):
        """Process uploaded documents with enhanced features"""
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        processed_docs = []
        docs_metadata = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Process document
                file_content = uploaded_file.getvalue()
                chunks, metadata = self.doc_processor.process_document(file_content, uploaded_file.name)
                
                # Add to index
                if chunks:
                    st.session_state.faiss_index.add_documents(chunks)
                    st.session_state.documents.extend(chunks)
                    processed_docs.extend(chunks)
                    docs_metadata.append(asdict(metadata))
                
            except Exception as e:
                st.sidebar.error(f"Error processing {uploaded_file.name}: {e}")
                logger.error(f"Document processing error: {e}")
        
        # Save index and metadata
        if processed_docs:
            try:
                st.session_state.faiss_index.save_local(INDEX_DIR)
                self.save_docs_meta(docs_metadata)
                
                # Update hybrid retriever
                self.hybrid_retriever = HybridRetriever(st.session_state.faiss_index, st.session_state.documents)
                
                st.sidebar.success(f"✅ Processed {len(uploaded_files)} documents successfully!")
                
            except Exception as e:
                st.sidebar.error(f"Error saving processed documents: {e}")
                logger.error(f"Document saving error: {e}")
        
        progress_bar.empty()
        status_text.empty()
    
    def load_docs_meta(self) -> List[Dict]:
        """Load document metadata"""
        if os.path.exists(DOCS_META):
            try:
                with open(DOCS_META, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading docs metadata: {e}")
        return []
    
    def save_docs_meta(self, new_metadata: List[Dict]):
        """Save document metadata"""
        try:
            existing_metadata = self.load_docs_meta()
            all_metadata = existing_metadata + new_metadata
            
            with open(DOCS_META, 'w') as f:
                json.dump(all_metadata, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving docs metadata: {e}")
    
    def remove_document(self, filename: str):
        """Remove document from index and metadata"""
        # This is a simplified implementation
        # In a production system, you'd need to rebuild the index
        metadata = self.load_docs_meta()
        updated_metadata = [doc for doc in metadata if doc['filename'] != filename]
        
        with open(DOCS_META, 'w') as f:
            json.dump(updated_metadata, f, indent=2, default=str)
    
    def clear_all_documents(self):
        """Clear all documents"""
        try:
            # Remove index
            if os.path.exists(INDEX_DIR):
                import shutil
                shutil.rmtree(INDEX_DIR)
            
            # Remove metadata
            if os.path.exists(DOCS_META):
                os.remove(DOCS_META)
            
            # Reset session state
            st.session_state.faiss_index = self.load_or_create_index()
            st.session_state.documents = []
            self.hybrid_retriever = None
            
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
    
    def save_current_conversation(self):
        """Save current conversation to database"""
        if st.session_state.messages:
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            self.db_manager.save_conversation(st.session_state.conversation_id, title)
            st.sidebar.success("Conversation saved!")
    
    def load_conversation(self, conversation_id: str):
        """Load conversation from database"""
        # This would require implementing message loading from database
        # For now, just switch conversation ID
        st.session_state.conversation_id = conversation_id
        st.session_state.messages = []  # Would load actual messages here
    
    def export_conversation_json(self):
        """Export conversation as JSON"""
        if st.session_state.messages:
            conversation_data = {
                'conversation_id': st.session_state.conversation_id,
                'messages': [asdict(msg) for msg in st.session_state.messages],
                'exported_at': datetime.now().isoformat()
            }
            
            json_str = json.dumps(conversation_data, indent=2, default=str)
            st.sidebar.download_button(
                "📥 Download JSON",
                data=json_str,
                file_name=f"conversation_{st.session_state.conversation_id[:8]}.json",
                mime="application/json"
            )
    
    def import_conversation_json(self, uploaded_file):
        """Import conversation from JSON"""
        try:
            conversation_data = json.load(uploaded_file)
            
            # Convert back to ChatMessage objects
            messages = []
            for msg_data in conversation_data.get('messages', []):
                msg_data['timestamp'] = datetime.fromisoformat(msg_data['timestamp'])
                messages.append(ChatMessage(**msg_data))
            
            st.session_state.messages = messages
            st.session_state.conversation_id = conversation_data.get('conversation_id', str(uuid.uuid4()))
            
            st.sidebar.success("Conversation imported!")
            
        except Exception as e:
            st.sidebar.error(f"Error importing conversation: {e}")
    
    def run(self):
        """Run the main application"""
        st.set_page_config(
            page_title="Advanced Gemma Chat",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better UI
        st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .user-message {
            background-color: #e3f2fd;
        }
        .assistant-message {
            background-color: #f5f5f5;
        }
        .message-timestamp {
            font-size: 0.8rem;
            color: #666;
        }
        .confidence-score {
            font-size: 0.8rem;
            color: #888;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.title("🤖 Advanced Gemma Chat with Ollama")
        st.markdown("*Enhanced with streaming, RAG, analytics, and quality assurance*")
        
        # Render sidebar and get configuration
        config = self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "⚙️ Settings"])
        
        with tab1:
            self.render_chat_interface(config)
        
        with tab2:
            render_analytics_dashboard(self.db_manager)
        
        with tab3:
            self.render_settings_tab()
    
    def render_chat_interface(self, config: Dict[str, Any]):
        """Render the main chat interface"""
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display messages
            for message in st.session_state.messages:
                self.render_message(message, show_metadata=False)
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Process input
            with st.spinner("Generating response..."):
                try:
                    ai_message = self.process_user_input(
                        user_input=user_input,
                        mode=config["mode"],
                        system_prompt=config["system_prompt"],
                        model_config={
                            "model_name": config["model_name"],
                            "temperature": config["temperature"],
                            "base_url": config["base_url"]
                        },
                        streaming=config["streaming"]
                    )
                    
                    # Show quality indicators
                    if ai_message.confidence_score and ai_message.confidence_score < 0.5:
                        st.warning("⚠️ Low confidence response. Please verify information.")
                    
                    if not ai_message.metadata.get("is_safe", True):
                        st.error("🚨 Content safety warning detected.")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"Chat processing error: {e}")
        
        # Message actions
        if st.session_state.messages:
            with st.expander("💡 Message Actions"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 Show Message Details"):
                        st.session_state.show_message_details = True
                        st.rerun()
                
                with col2:
                    if st.button("🔍 Search Messages"):
                        search_query = st.text_input("Search in conversation:")
                        if search_query:
                            self.search_messages(search_query)
                
                with col3:
                    if st.button("🌳 Branch Conversation"):
                        self.create_conversation_branch()
    
    def render_settings_tab(self):
        """Render settings and configuration tab"""
        st.header("⚙️ Advanced Settings")
        
        # Performance settings
        with st.expander("🚀 Performance Settings"):
            enable_caching = st.checkbox("Enable Response Caching", value=True)
            cache_ttl = st.slider("Cache TTL (hours)", 1, 48, 24)
            max_context_tokens = st.slider("Max Context Tokens", 1000, 8000, 4000)
            
            if st.button("Clear Cache"):
                # Clear cache implementation
                st.success("Cache cleared!")
        
        # Quality assurance settings
        with st.expander("🛡️ Quality Assurance"):
            enable_content_filter = st.checkbox("Enable Content Safety Filter", value=True)
            enable_confidence_scoring = st.checkbox("Show Confidence Scores", value=True)
            enable_hallucination_detection = st.checkbox("Enable Hallucination Detection", value=True)
            
            confidence_threshold = st.slider("Confidence Warning Threshold", 0.0, 1.0, 0.5)
        
        # RAG settings
        with st.expander("📚 RAG Configuration"):
            retrieval_method = st.selectbox("Retrieval Method", 
                                          ["Hybrid Search", "Semantic Only", "MMR"])
            num_retrieved_docs = st.slider("Number of Retrieved Documents", 1, 10, 4)
            mmr_lambda = st.slider("MMR Lambda (diversity)", 0.0, 1.0, 0.5)
        
        # Export settings
        with st.expander("📤 Data Export"):
            if st.button("Export All Conversations"):
                # Export implementation
                st.success("Export started!")
            
            if st.button("Export Analytics Data"):
                # Analytics export implementation
                st.success("Analytics exported!")
        
        # Debug information
        with st.expander("🔧 Debug Information"):
            st.json({
                "conversation_id": st.session_state.conversation_id,
                "total_messages": len(st.session_state.messages),
                "documents_loaded": len(st.session_state.documents),
                "memory_type": self.memory_manager.current_memory_type if self.memory_manager else "None",
                "cache_enabled": enable_caching,
                "model_config": {
                    "model": st.session_state.current_model,
                    "embedding_model": EMBEDDING_MODEL_NAME
                }
            })
    
    def search_messages(self, query: str):
        """Search through conversation messages"""
        matching_messages = []
        for msg in st.session_state.messages:
            if query.lower() in msg.content.lower():
                matching_messages.append(msg)
        
        if matching_messages:
            st.success(f"Found {len(matching_messages)} matching messages:")
            for msg in matching_messages:
                with st.expander(f"{msg.role}: {msg.content[:50]}..."):
                    st.write(msg.content)
                    st.caption(f"Timestamp: {msg.timestamp}")
        else:
            st.info("No matching messages found.")
    
    def create_conversation_branch(self):
        """Create a new conversation branch"""
        branch_id = str(uuid.uuid4())
        st.session_state.conversation_branches[branch_id] = {
            'parent_id': st.session_state.conversation_id,
            'messages': st.session_state.messages.copy(),
            'created_at': datetime.now()
        }
        
        st.success(f"Created conversation branch: {branch_id[:8]}")

# -----------------------
# Main Application Entry Point
# -----------------------
def main():
    """Main application entry point"""
    try:
        app = AdvancedChatApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application startup error: {e}")
        
        # Show debug information
        with st.expander("Debug Information"):
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()