"""
Streamlit GUI for Local RAG (WSL-only Ollama) — Enhanced Version
================================================================
A Retrieval-Augmented Generation system for local document processing.

Key Features:
- GPU-aware CrossEncoder reranking with caching and batching
- WSL-optimized Ollama integration with connection validation
- Robust document processing with multiple file format support
- Advanced retrieval strategies (MMR, reranking, HyDE)
- DeepConf confidence-aware reasoning for improved accuracy
- Comprehensive metrics and monitoring

Dependencies:
    Required:
        - streamlit
        - langchain-community
        - langchain-text-splitters
        - chromadb
        - pypdf
        - requests
    Optional:
        - torch (for GPU acceleration)
        - sentence-transformers (for reranking)
        - unstructured (for markdown processing)
"""

import os
import logging
import json
import requests
import tempfile
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from collections import deque, Counter
from statistics import mean
from functools import lru_cache

import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.schema import Document

# Optional heavy dependencies - resolved at runtime to avoid import errors
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# ====================================================================
# LOGGING CONFIGURATION
# ====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ====================================================================
# CONFIGURATION MANAGEMENT
# ====================================================================
class Config:
    """
    Centralized configuration management for the RAG system.
    
    This class holds all configurable parameters and provides
    validation utilities for system components.
    """
    
    # Vector database configuration
    DEFAULT_DB_DIR = "chroma_db"
    
    # Model configuration
    DEFAULT_LLM_MODEL = "llama3.1:8b"
    DEFAULT_EMBED_MODEL = "nomic-embed-text"
    
    # Document processing configuration
    DEFAULT_CHUNK_SIZE = 800  # Characters per chunk
    DEFAULT_CHUNK_OVERLAP = 120  # Overlap between chunks
    
    # Generation configuration
    DEFAULT_TEMPERATURE = 0.2  # Lower = more deterministic
    
    # Retrieval configuration
    DEFAULT_TOP_K = 8  # Documents to retrieve
    DEFAULT_RERANK_TOP_K = 5  # Documents after reranking
    
    # Ollama connection configuration
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    
    # File processing limits
    MAX_FILE_SIZE_MB = 50
    SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".csv", ".json"]
    
    @classmethod
    def validate_ollama_connection(cls) -> Tuple[bool, str]:
        """
        Validate connection to Ollama service.
        
        Returns:
            Tuple[bool, str]: (success_status, message)
        """
        try:
            response = requests.get(
                f"{cls.OLLAMA_HOST}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                return True, "Connected to Ollama"
            return False, f"Ollama returned status {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"Cannot connect to Ollama: {e}"


# ====================================================================
# DOCUMENT PROCESSING
# ====================================================================
class DocumentProcessor:
    """
    Enhanced document processing with validation, deduplication, and chunking.
    
    This class handles the entire document ingestion pipeline from
    file upload to chunk creation.
    """
    
    @staticmethod
    def get_file_hash(file_content: bytes) -> str:
        """
        Generate MD5 hash for file deduplication.
        
        Args:
            file_content: Raw file bytes
            
        Returns:
            str: Hexadecimal MD5 hash
        """
        return hashlib.md5(file_content).hexdigest()
    
    @staticmethod
    def validate_file(uploaded_file) -> Tuple[bool, str]:
        """
        Validate uploaded file for size and type constraints.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple[bool, str]: (is_valid, validation_message)
        """
        # Check file size
        size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if size_mb > Config.MAX_FILE_SIZE_MB:
            return False, f"File too large: {size_mb:.1f}MB (max {Config.MAX_FILE_SIZE_MB}MB)"
        
        # Check file extension
        ext = Path(uploaded_file.name).suffix.lower()
        if ext not in Config.SUPPORTED_EXTENSIONS:
            return False, f"Unsupported file type: {ext}"
        
        return True, "Valid"
    
    @staticmethod
    def load_document(file_path: str, file_type: str) -> List[Document]:
        """
        Load document using appropriate loader based on file type.
        
        Args:
            file_path: Path to temporary file
            file_type: File extension (e.g., '.pdf')
            
        Returns:
            List[Document]: Loaded document chunks
            
        Raises:
            ValueError: If no loader available for file type
        """
        # Map file types to their respective loaders
        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": lambda p: TextLoader(p, encoding="utf-8"),
            ".md": lambda p: UnstructuredMarkdownLoader(p, mode="elements"),
            ".json": lambda p: TextLoader(p, encoding="utf-8"),
            ".csv": lambda p: TextLoader(p, encoding="utf-8"),
        }
        
        loader_cls = loaders.get(file_type)
        if not loader_cls:
            raise ValueError(f"No loader for file type: {file_type}")
        
        loader = loader_cls(file_path)
        return loader.load()
    
    @classmethod
    def process_uploads(
        cls,
        uploaded_files: List[Any],
        chunk_size: int,
        chunk_overlap: int,
    ) -> Tuple[List[Document], List[str], int]:
        """
        Process multiple uploaded files with validation, deduplication, and chunking.
        
        Args:
            uploaded_files: List of Streamlit uploaded file objects
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            Tuple containing:
                - List[Document]: Processed document chunks
                - List[str]: Error messages for failed files
                - int: Count of unique files processed
        """
        all_chunks: List[Document] = []
        errors: List[str] = []
        processed_hashes = set()
        unique_files = 0
        
        for uploaded in uploaded_files:
            try:
                # Validate file
                is_valid, validation_msg = cls.validate_file(uploaded)
                if not is_valid:
                    errors.append(f"{uploaded.name}: {validation_msg}")
                    continue
                
                # Check for duplicates using file hash
                file_hash = cls.get_file_hash(uploaded.getvalue())
                if file_hash in processed_hashes:
                    errors.append(f"{uploaded.name}: Duplicate file skipped")
                    continue
                
                processed_hashes.add(file_hash)
                unique_files += 1
                
                # Create temporary file for processing
                suffix = Path(uploaded.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                
                try:
                    # Load document with appropriate loader
                    try:
                        docs = cls.load_document(tmp_path, suffix)
                    except Exception as e:
                        # Fallback for Markdown if unstructured is not installed
                        if suffix == ".md":
                            logger.info("Falling back to TextLoader for Markdown")
                            docs = TextLoader(tmp_path, encoding="utf-8").load()
                        else:
                            raise
                    
                    # Enrich metadata for each document
                    for doc in docs:
                        doc.metadata.update({
                            "source": uploaded.name,
                            "file_hash": file_hash,
                            "upload_time": datetime.now().isoformat(),
                            "file_type": suffix,
                        })
                    
                    # Split documents into chunks
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                        length_function=len,
                    )
                    chunks = splitter.split_documents(docs)
                    all_chunks.extend(chunks)
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass  # Non-critical error
                        
            except Exception as e:
                error_msg = f"{uploaded.name}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        return all_chunks, errors, unique_files


# ====================================================================
# RETRIEVAL ENGINE
# ====================================================================
@st.cache_resource(show_spinner=False)
def get_reranker_model():
    """
    Initialize and cache the CrossEncoder reranking model.
    
    Uses GPU if available, falls back to CPU otherwise.
    
    Returns:
        CrossEncoder model or None if initialization fails
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        logger.warning(f"sentence-transformers not available: {e}")
        return None
    
    # Determine device (GPU/CPU)
    device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    logger.info(f"Initializing reranker on device: {device}")
    
    try:
        return CrossEncoder("BAAI/bge-reranker-base", device=device)
    except Exception as e:
        logger.warning(f"Reranker initialization failed: {e}")
        return None


class EnhancedRetriever:
    """
    Advanced retrieval system with MMR diversity and optional reranking.
    
    This class implements sophisticated retrieval strategies including:
    - Maximum Marginal Relevance (MMR) for diversity
    - Cross-encoder reranking for improved precision
    - Hybrid search combining different strategies
    """
    
    def __init__(self, vectorstore: Chroma):
        """
        Initialize retriever with vector store.
        
        Args:
            vectorstore: Chroma vector database instance
        """
        self.vectorstore = vectorstore
        self._reranker = None
    
    @property
    def reranker(self):
        """Lazy-load reranker model."""
        if self._reranker is None:
            self._reranker = get_reranker_model()
        return self._reranker
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        use_mmr: bool = True,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """
        Perform hybrid search with optional MMR diversity.
        
        Args:
            query: Search query string
            k: Number of documents to retrieve
            use_mmr: Whether to use Maximum Marginal Relevance
            lambda_mult: Balance between relevance and diversity (0-1)
            
        Returns:
            List[Document]: Retrieved documents
        """
        if use_mmr:
            # Use MMR for diverse retrieval
            docs = self.vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                lambda_mult=lambda_mult
            )
        else:
            # Standard similarity search
            pairs = self.vectorstore.similarity_search_with_score(query, k=k)
            docs = [doc for doc, score in pairs]
        
        return docs
    
    def rerank_documents(
        self,
        query: str,
        docs: List[Document],
        top_k: int
    ) -> List[Document]:
        """
        Rerank documents using cross-encoder model.
        
        Args:
            query: Original search query
            docs: Documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List[Document]: Reranked documents
        """
        if not self.reranker or not docs:
            return docs[:top_k]
        
        try:
            # Prepare query-document pairs for reranking
            pairs = [[query, doc.page_content] for doc in docs]
            
            # Get reranking scores (batched for efficiency)
            scores = self.reranker.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False
            )
            
            # Sort by score and return top-k
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in ranked[:top_k]]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return docs[:top_k]
    
    def retrieve(
        self,
        query: str,
        k: int = 8,
        use_reranker: bool = False,
        rerank_top_k: int = 5,
        use_mmr: bool = True
    ) -> List[Document]:
        """
        Main retrieval method combining all strategies.
        
        Args:
            query: Search query
            k: Initial documents to retrieve
            use_reranker: Whether to apply reranking
            rerank_top_k: Final number of documents after reranking
            use_mmr: Whether to use MMR for diversity
            
        Returns:
            List[Document]: Final retrieved documents
        """
        # Initial retrieval
        docs = self.hybrid_search(query, k=k, use_mmr=use_mmr)
        
        # Optional reranking for precision
        if use_reranker and len(docs) > rerank_top_k:
            docs = self.rerank_documents(query, docs, rerank_top_k)
        
        return docs


# ====================================================================
# DEEPCONF: CONFIDENCE-AWARE REASONING
# ====================================================================
class DeepConf:
    """
    DeepConf implementation for confidence-aware text generation.
    
    This class implements advanced reasoning techniques that use
    token-level confidence scores to improve generation quality
    through multiple sampling and consensus mechanisms.
    """
    
    def __init__(
        self,
        base_url: str,
        model_name: str,
        topk: int = 5,
        group_size: int = 128
    ):
        """
        Initialize DeepConf with model configuration.
        
        Args:
            base_url: Ollama API endpoint
            model_name: Name of the LLM model
            topk: Number of top tokens to consider for confidence
            group_size: Window size for confidence calculation
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.topk = max(1, int(topk))
        self.group_size = max(8, int(group_size))
    
    def stream_tokens(
        self,
        prompt: str,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Stream tokens from Ollama with logprobs for confidence calculation.
        
        Args:
            prompt: Input prompt text
            options: Additional generation options
            
        Yields:
            Dict containing token, logprobs, and completion status
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {"logprobs": self.topk, **(options or {})}
        }
        
        try:
            with requests.post(url, json=payload, stream=True, timeout=600) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        event = json.loads(line.decode("utf-8", errors="ignore"))
                    except json.JSONDecodeError:
                        continue
                    
                    # Extract token and logprobs
                    token = event.get("response") or event.get("token")
                    if token is None:
                        continue
                    
                    # Parse logprobs from different possible formats
                    lp_list = self._extract_logprobs_list(event)
                    single_lp = self._extract_single_logprob(event)
                    
                    yield {
                        "token": token,
                        "logprobs": lp_list,
                        "logprob": single_lp,
                        "done": bool(event.get("done"))
                    }
                    
        except Exception as e:
            logger.error(f"DeepConf stream error: {e}")
            return
    
    def _extract_logprobs_list(self, event: Dict) -> Optional[List[float]]:
        """Extract top-k logprobs from event data."""
        # Try different formats Ollama might use
        if isinstance(event.get("logprobs"), dict):
            top = event["logprobs"].get("top_logprobs")
            if isinstance(top, list):
                vals = [
                    float(item["logprob"])
                    for item in top
                    if isinstance(item, dict) and "logprob" in item
                ]
                return vals[:self.topk] if vals else None
                
        elif isinstance(event.get("top_logprobs"), list):
            vals = [
                float(item["logprob"])
                for item in event["top_logprobs"]
                if isinstance(item, dict) and "logprob" in item
            ]
            return vals[:self.topk] if vals else None
        
        return None
    
    def _extract_single_logprob(self, event: Dict) -> Optional[float]:
        """Extract single logprob value from event data."""
        if "logprob" in event and isinstance(event["logprob"], (int, float)):
            return float(event["logprob"])
        return None
    
    @staticmethod
    def token_confidence_from_logprobs(
        topk_logprobs: Optional[List[float]],
        single_logprob: Optional[float]
    ) -> Optional[float]:
        """
        Calculate confidence score from logprobs.
        
        Higher confidence = lower uncertainty.
        
        Args:
            topk_logprobs: List of top-k log probabilities
            single_logprob: Single log probability value
            
        Returns:
            float: Confidence score (higher is better)
        """
        if topk_logprobs and len(topk_logprobs) > 0:
            # Use negative mean of top-k logprobs as confidence
            return -mean(topk_logprobs)
        if isinstance(single_logprob, (int, float)):
            return -float(single_logprob)
        return None
    
    def compute_trace_metrics(
        self,
        token_events: List[Dict[str, Any]],
        tail_window: int = 128
    ) -> Dict[str, Any]:
        """
        Compute confidence metrics for a generation trace.
        
        Args:
            token_events: List of token generation events
            tail_window: Size of tail window for metrics
            
        Returns:
            Dict containing various confidence metrics
        """
        # Extract confidence scores for all tokens
        confidences: List[float] = []
        for event in token_events:
            conf = self.token_confidence_from_logprobs(
                event.get("logprobs"),
                event.get("logprob")
            )
            if conf is not None:
                confidences.append(conf)
        
        if not confidences:
            return {
                "per_token": [],
                "avg": None,
                "bottom10": None,
                "lowest_group": None,
                "tail": None
            }
        
        # Calculate group-based metrics using sliding window
        groups = self._calculate_group_metrics(confidences)
        
        # Calculate percentile-based metrics
        sorted_groups = sorted(groups)
        k = max(1, int(0.10 * len(sorted_groups)))  # Bottom 10%
        bottom10_mean = mean(sorted_groups[:k])
        lowest_group = sorted_groups[0]
        
        # Calculate tail metric (last N tokens)
        if len(confidences) >= tail_window:
            tail_mean = mean(confidences[-tail_window:])
        else:
            tail_mean = mean(confidences)
        
        return {
            "per_token": confidences,
            "avg": mean(confidences),
            "bottom10": bottom10_mean,
            "lowest_group": lowest_group,
            "tail": tail_mean
        }
    
    def _calculate_group_metrics(self, confidences: List[float]) -> List[float]:
        """Calculate sliding window group metrics."""
        if len(confidences) >= self.group_size:
            # Use sliding window for group calculation
            window = deque(confidences[:self.group_size], maxlen=self.group_size)
            groups = [mean(window)]
            
            for conf in confidences[self.group_size:]:
                window.append(conf)
                groups.append(mean(window))
        else:
            # Single group if not enough tokens
            groups = [mean(confidences)]
        
        return groups
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        """
        Normalize answer text for comparison.
        
        Args:
            text: Raw answer text
            
        Returns:
            str: Normalized text (lowercased, whitespace collapsed)
        """
        return " ".join(text.strip().lower().split())
    
    @staticmethod
    def weighted_vote(candidates: List[Tuple[str, float]]) -> Tuple[str, Dict[str, float]]:
        """
        Perform weighted voting on candidate answers.
        
        Args:
            candidates: List of (answer, confidence) tuples
            
        Returns:
            Tuple of (best_answer, vote_tally)
        """
        tally: Dict[str, float] = {}
        
        for answer, weight in candidates:
            key = DeepConf.normalize_answer(answer)
            tally[key] = tally.get(key, 0.0) + float(weight or 0.0)
        
        if not tally:
            return "", {}
        
        best = max(tally.items(), key=lambda kv: kv[1])[0]
        return best, tally
    
    def offline_decide(
        self,
        prompt: str,
        K: int = 16,
        metric: str = "bottom10",
        keep_ratio: float = 0.9,
        temperature: float = 0.7,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Offline decision mode: Generate K traces and select based on confidence.
        
        Args:
            prompt: Input prompt
            K: Number of traces to generate
            metric: Confidence metric to use
            keep_ratio: Ratio of high-confidence traces to keep
            temperature: Generation temperature
            options: Additional generation options
            
        Returns:
            Dict containing answer, traces, and metadata
        """
        traces = []
        
        # Generate K independent traces
        for _ in range(K):
            events, parts = [], []
            
            for event in self.stream_tokens(
                prompt,
                options={"temperature": temperature, **(options or {})}
            ):
                events.append(event)
                parts.append(event["token"] or "")
                if event.get("done"):
                    break
            
            text = "".join(parts).strip()
            metrics = self.compute_trace_metrics(events)
            conf = metrics.get(metric) if metrics else None
            
            traces.append({
                "text": text,
                "metrics": metrics,
                "conf": conf
            })
        
        # Filter and vote based on confidence
        confidences = [
            t["conf"] for t in traces
            if isinstance(t.get("conf"), (int, float))
        ]
        
        if not confidences:
            # No confidence available, equal weight voting
            best, tally = self.weighted_vote([(t["text"], 1.0) for t in traces])
            return {
                "answer": best,
                "traces": traces,
                "tally": tally,
                "used_metric": None,
                "kept": len(traces)
            }
        
        # Sort by confidence and keep top ratio
        sorted_traces = sorted(
            [t for t in traces if t["conf"] is not None],
            key=lambda x: x["conf"],
            reverse=True
        )
        keep_n = max(1, int(round(keep_ratio * len(sorted_traces))))
        kept = sorted_traces[:keep_n]
        
        # Weighted vote using confidence scores
        best, tally = self.weighted_vote([(t["text"], t["conf"]) for t in kept])
        
        return {
            "answer": best,
            "traces": traces,
            "tally": tally,
            "used_metric": metric,
            "kept": keep_n
        }
    
    def online_decide(
        self,
        prompt: str,
        budget: int = 64,
        Ninit: int = 8,
        mode: str = "high",
        metric: str = "lowest_group",
        consensus_tau: float = 0.95,
        temperature: float = 0.7,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Online decision mode: Adaptive generation with early stopping.
        
        This mode starts with initial traces, establishes a confidence
        threshold, then generates additional traces with early stopping
        based on confidence monitoring.
        
        Args:
            prompt: Input prompt
            budget: Maximum number of traces
            Ninit: Number of initial warmup traces
            mode: "high" for high-confidence or "low" for exploration
            metric: Confidence metric to use
            consensus_tau: Consensus threshold for early stopping
            temperature: Generation temperature
            options: Additional generation options
            
        Returns:
            Dict containing answer, traces, and metadata
        """
        # Warmup phase: generate initial traces
        warmup = self.offline_decide(
            prompt=prompt,
            K=Ninit,
            metric=metric,
            keep_ratio=0.9 if mode == "high" else 0.1,
            temperature=temperature,
            options=options
        )
        
        # Extract confidence values from warmup
        kept_conf = [
            t["conf"] for t in warmup["traces"]
            if t["conf"] is not None
        ]
        
        if not kept_conf:
            # Fallback if no confidence available
            fallback = self.offline_decide(
                prompt=prompt,
                K=budget,
                metric="avg",
                keep_ratio=1.0,
                temperature=temperature,
                options=options
            )
            return {**fallback, "threshold": None, "mode": "fallback_no_conf"}
        
        # Determine confidence threshold
        threshold = self._calculate_threshold(kept_conf, mode)
        
        # Continue from warmup traces
        traces = warmup["traces"][:]
        best, tally = self.weighted_vote([
            (t["text"], float(t["conf"] or 0.0))
            for t in traces
        ])
        
        # Check initial consensus
        if self._check_consensus(tally, consensus_tau) or len(traces) >= budget:
            return {
                "answer": best,
                "traces": traces,
                "tally": tally,
                "threshold": threshold,
                "mode": mode
            }
        
        # Online phase: generate with early stopping
        while len(traces) < budget:
            trace = self._generate_with_early_stopping(
                prompt,
                threshold,
                temperature,
                options
            )
            traces.append(trace)
            
            # Update voting
            best, tally = self.weighted_vote([
                (t["text"], float(t["conf"] or 0.0))
                for t in traces
            ])
            
            # Check for consensus
            if self._check_consensus(tally, consensus_tau):
                break
        
        return {
            "answer": best,
            "traces": traces,
            "tally": tally,
            "threshold": threshold,
            "mode": mode
        }
    
    def _calculate_threshold(self, confidences: List[float], mode: str) -> float:
        """Calculate confidence threshold based on mode."""
        sorted_conf = sorted(confidences)
        
        if mode == "low":
            # High threshold for exploration mode
            idx = min(len(sorted_conf) - 1, int(0.9 * (len(sorted_conf) - 1)))
        else:
            # Low threshold for high-confidence mode
            idx = min(len(sorted_conf) - 1, int(0.1 * (len(sorted_conf) - 1)))
        
        return sorted_conf[idx]
    
    def _check_consensus(self, tally: Dict[str, float], tau: float) -> bool:
        """Check if consensus threshold is reached."""
        if not tally:
            return False
        
        total = sum(tally.values()) or 1.0
        top = max(tally.values())
        consensus_ratio = top / total
        
        return consensus_ratio >= tau
    
    def _generate_with_early_stopping(
        self,
        prompt: str,
        threshold: float,
        temperature: float,
        options: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate trace with early stopping based on confidence."""
        events, parts = [], []
        token_confs = []
        window = deque(maxlen=self.group_size)
        stopped_early = False
        
        for event in self.stream_tokens(
            prompt,
            options={"temperature": temperature, **(options or {})}
        ):
            events.append(event)
            parts.append(event["token"] or "")
            
            # Track confidence
            conf = self.token_confidence_from_logprobs(
                event.get("logprobs"),
                event.get("logprob")
            )
            
            if conf is not None:
                token_confs.append(conf)
                window.append(conf)
                
                # Check for early stopping
                if len(window) == self.group_size:
                    group_conf = mean(window)
                    if group_conf < threshold:
                        stopped_early = True
                        break
            
            if event.get("done"):
                break
        
        text = "".join(parts).strip()
        metrics = self.compute_trace_metrics(events)
        conf = metrics.get("lowest_group") if metrics else None
        
        return {
            "text": text,
            "metrics": metrics,
            "conf": conf,
            "stopped": stopped_early
        }


# ====================================================================
# PROMPT TEMPLATES
# ====================================================================
class PromptTemplates:
    """
    Collection of prompt templates for different interaction styles.
    
    Each template is designed to elicit different types of responses
    from the LLM while maintaining citation requirements.
    """
    
    # Basic factual Q&A template
    BASIC_RAG = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly.
Include citations [1], [2], etc. that correspond to the source documents.

Question: {question}

Context:
{context}

Answer:"""
    )
    
    # Analytical template for detailed analysis
    ANALYTICAL_RAG = ChatPromptTemplate.from_template(
        """You are an analytical assistant. Based on the provided context:
1. Answer the question comprehensively
2. Highlight any contradictions or uncertainties in the sources
3. Provide citations [1], [2], etc. for each claim

Question: {question}

Context:
{context}

Analysis:"""
    )
    
    # Creative template for engaging responses
    CREATIVE_RAG = ChatPromptTemplate.from_template(
        """Based on the context provided, give a detailed and engaging answer.
Explain concepts thoroughly and make connections between ideas.
Always cite your sources with [1], [2], etc.

Question: {question}

Context:
{context}

Response:"""
    )


# ====================================================================
# CACHED RESOURCE MANAGEMENT
# ====================================================================
@st.cache_resource(show_spinner=False)
def get_embeddings(embed_model: str) -> OllamaEmbeddings:
    """
    Get cached embedding model instance.
    
    Args:
        embed_model: Name of the embedding model
        
    Returns:
        OllamaEmbeddings: Cached embeddings instance
    """
    return OllamaEmbeddings(
        model=embed_model,
        base_url=Config.OLLAMA_HOST
    )


@st.cache_resource(show_spinner=False)
def get_vectorstore(persist_directory: str, embed_model: str) -> Chroma:
    """
    Get or create cached vector store instance.
    
    Args:
        persist_directory: Directory for vector store persistence
        embed_model: Name of the embedding model
        
    Returns:
        Chroma: Cached vector store instance
    """
    # Ensure directory exists
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    
    # Get embedding function
    embeddings = get_embeddings(embed_model)
    
    # Create or load vector store
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


@st.cache_resource(show_spinner=False)
def get_llm(model: str, temperature: float) -> Ollama:
    """
    Get cached LLM instance with specified configuration.
    
    Args:
        model: Name of the LLM model
        temperature: Generation temperature
        
    Returns:
        Ollama: Cached LLM instance
    """
    return Ollama(
        model=model,
        temperature=temperature,
        base_url=Config.OLLAMA_HOST,
        num_ctx=4096,        # Context window size
        num_predict=1024,    # Max tokens to generate
        top_p=0.9,          # Nucleus sampling threshold
        repeat_penalty=1.1,  # Repetition penalty
    )


# ====================================================================
# SESSION STATE MANAGEMENT
# ====================================================================
def init_session_state():
    """
    Initialize Streamlit session state variables.
    
    This function ensures all required session state variables
    are initialized with appropriate default values.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict[str, Any]] = []
    
    if "files_ingested" not in st.session_state:
        st.session_state.files_ingested = 0
    
    if "chunks_ingested" not in st.session_state:
        st.session_state.chunks_ingested = 0
    
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0


# ====================================================================
# USER INTERFACE: SIDEBAR
# ====================================================================
def render_sidebar() -> Dict[str, Any]:
    """
    Render the configuration sidebar and return settings.
    
    This function creates the sidebar UI with all configuration
    options and returns the current settings as a dictionary.
    
    Returns:
        Dict[str, Any]: Current configuration settings
    """
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.caption(f"Ollama endpoint: {Config.OLLAMA_HOST}")
        
        # Connection status
        ok, msg = Config.validate_ollama_connection()
        if ok:
            st.success(f"✅ {msg}")
        else:
            st.error(f"❌ {msg}")
            st.info("Ensure Ollama is running in WSL and reachable via OLLAMA_HOST")
        
        # Model settings
        st.subheader("🤖 Model Settings")
        model_name = st.text_input(
            "LLM Model",
            value=Config.DEFAULT_LLM_MODEL,
            help="Name of the Ollama model for generation"
        )
        embed_name = st.text_input(
            "Embedding Model",
            value=Config.DEFAULT_EMBED_MODEL,
            help="Name of the Ollama model for embeddings"
        )
        temperature = st.slider(
            "Temperature",
            0.0, 1.0, Config.DEFAULT_TEMPERATURE, 0.1,
            help="Higher values make output more random"
        )
        
        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        top_k = st.slider(
            "Documents to retrieve",
            1, 30, Config.DEFAULT_TOP_K,
            help="Number of documents to retrieve from vector store"
        )
        use_mmr = st.checkbox(
            "Use MMR for diversity",
            value=True,
            help="Maximum Marginal Relevance balances relevance and diversity"
        )
        use_reranker = st.checkbox(
            "Use reranker (GPU if available)",
            value=False,
            help="Cross-encoder reranking for improved precision"
        )
        rerank_top_k = Config.DEFAULT_RERANK_TOP_K
        if use_reranker:
            rerank_top_k = st.slider(
                "Rerank to top-k",
                1, 20, Config.DEFAULT_RERANK_TOP_K,
                help="Final number of documents after reranking"
            )
        
        # Prompt style selection
        st.subheader("📝 Prompt Style")
        prompt_style = st.selectbox(
            "Select prompt style",
            ["Basic", "Analytical", "Creative"],
            index=0,
            help="Different styles produce different types of responses"
        )
        
        # DeepConf settings
        st.subheader("🧠 Reasoning (DeepConf)")
        enable_deepconf = st.checkbox(
            "Enable DeepConf",
            value=False,
            help="Use confidence-aware reasoning for improved accuracy"
        )
        
        deepconf_settings = {}
        if enable_deepconf:
            deepconf_mode = st.selectbox(
                "Mode",
                ["Offline", "Online"],
                index=1,
                help="Offline: Generate all traces upfront. Online: Adaptive generation"
            )
            deepconf_metric = st.selectbox(
                "Confidence metric",
                ["bottom10", "lowest_group", "tail", "avg"],
                index=0,
                help="Metric for measuring generation confidence"
            )
            deepconf_budget = st.slider(
                "Trace budget (K)",
                4, 512, 32, 4,
                help="Maximum number of generation traces"
            )
            deepconf_keep = st.selectbox(
                "Filter keep ratio η",
                ["90%", "10%"],
                index=0,
                help="Percentage of high-confidence traces to keep"
            )
            deepconf_topk = st.slider(
                "Token top-k for confidence",
                1, 10, 5,
                help="Number of top tokens to consider for confidence"
            )
            deepconf_groupsize = st.slider(
                "Group size (tokens)",
                16, 2048, 128, 8,
                help="Window size for group confidence calculation"
            )
            
            if deepconf_mode == "Online":
                deepconf_tau = st.slider(
                    "Consensus threshold τ",
                    0.50, 1.00, 0.95, 0.01,
                    help="Threshold for early stopping based on consensus"
                )
                deepconf_ninit = st.slider(
                    "Warmup traces Ninit",
                    1, 64, 8,
                    help="Initial traces for threshold calibration"
                )
            else:
                deepconf_tau = 0.95
                deepconf_ninit = 8
            
            deepconf_settings = {
                "enable_deepconf": enable_deepconf,
                "deepconf_mode": deepconf_mode,
                "deepconf_metric": deepconf_metric,
                "deepconf_budget": deepconf_budget,
                "deepconf_keep": deepconf_keep,
                "deepconf_topk": deepconf_topk,
                "deepconf_groupsize": deepconf_groupsize,
                "deepconf_tau": deepconf_tau,
                "deepconf_ninit": deepconf_ninit,
            }
        
        # Retrieval helpers
        st.subheader("🔎 Retrieval Helpers")
        use_hyde = st.checkbox(
            "Expand query (HyDE-style)",
            value=False,
            help="Generate hypothetical answer to improve retrieval"
        )
        
        # Document filters
        st.subheader("🔍 Filters (optional)")
        filename_filter = st.text_input(
            "Filename contains",
            value="",
            help="Filter documents by filename substring"
        )
        filetype_filter = st.multiselect(
            "File types",
            options=[".pdf", ".txt", ".md", ".csv", ".json"],
            default=[],
            help="Filter documents by file type"
        )
        
        # Document processing settings
        st.subheader("📄 Document Processing")
        chunk_size = st.slider(
            "Chunk size",
            200, 2000, Config.DEFAULT_CHUNK_SIZE, 50,
            help="Size of text chunks in characters"
        )
        chunk_overlap = st.slider(
            "Chunk overlap",
            0, 400, Config.DEFAULT_CHUNK_OVERLAP, 10,
            help="Overlap between consecutive chunks"
        )
        
        # Vector store settings
        st.subheader("💾 Vector Store")
        db_dir = st.text_input(
            "Database directory",
            value=Config.DEFAULT_DB_DIR,
            help="Directory for vector store persistence"
        )
        
        # Statistics
        st.subheader("📊 Statistics")
        st.metric("Files Ingested", st.session_state.files_ingested)
        st.metric("Chunks Ingested", st.session_state.chunks_ingested)
        st.metric("Queries Processed", st.session_state.query_count)
        
        # Clear history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Combine all settings
        return {
            "model_name": model_name,
            "embed_name": embed_name,
            "temperature": temperature,
            "top_k": top_k,
            "use_mmr": use_mmr,
            "use_reranker": use_reranker,
            "rerank_top_k": rerank_top_k,
            "prompt_style": prompt_style,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "db_dir": db_dir,
            "use_hyde": use_hyde,
            "filename_filter": filename_filter,
            "filetype_filter": filetype_filter,
            **deepconf_settings
        }


# ====================================================================
# USER INTERFACE: DOCUMENT UPLOAD
# ====================================================================
def render_document_upload(settings: Dict[str, Any]):
    """
    Render the document upload interface.
    
    Args:
        settings: Current configuration settings
    """
    st.subheader("📁 Document Upload")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md", "csv", "json"],
            accept_multiple_files=True,
            help=f"Max file size: {Config.MAX_FILE_SIZE_MB}MB",
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        ingest_btn = st.button(
            "📥 Ingest Documents",
            type="primary",
            use_container_width=True
        )
    
    if ingest_btn and uploaded:
        with st.spinner("Processing documents…"):
            # Process uploaded files
            chunks, errors, unique_files = DocumentProcessor.process_uploads(
                uploaded,
                settings["chunk_size"],
                settings["chunk_overlap"]
            )
            
            # Add chunks to vector store
            if chunks:
                vs = get_vectorstore(settings["db_dir"], settings["embed_name"])
                vs.add_documents(chunks)
                
                # Update statistics
                st.session_state.files_ingested += unique_files
                st.session_state.chunks_ingested += len(chunks)
                
                st.success(f"✅ Ingested {unique_files} file(s), {len(chunks)} chunk(s)")
            
            # Display errors
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")


# ====================================================================
# USER INTERFACE: CHAT
# ====================================================================
def render_chat_interface(settings: Dict[str, Any]):
    """
    Render the main chat interface for Q&A.
    
    Args:
        settings: Current configuration settings
    """
    st.subheader("💬 Chat with your Documents")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for src in message["sources"]:
                        st.markdown(f"**[{src['id']}]** {src.get('source', 'Unknown')}")
                        st.code(src.get("preview", ""), language="text")
    
    # Chat input
    query = st.chat_input("Ask a question about your documents…")
    if not query:
        return
    
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.query_count += 1
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                # Get components
                vs = get_vectorstore(settings["db_dir"], settings["embed_name"])
                llm = get_llm(settings["model_name"], settings["temperature"])
                retriever = EnhancedRetriever(vs)
                
                # Select prompt template
                prompt_map = {
                    "Basic": PromptTemplates.BASIC_RAG,
                    "Analytical": PromptTemplates.ANALYTICAL_RAG,
                    "Creative": PromptTemplates.CREATIVE_RAG,
                }
                prompt = prompt_map[settings["prompt_style"]]
                
                # Prepare retrieval query (with optional HyDE expansion)
                retrieval_query = query
                if settings.get("use_hyde"):
                    try:
                        # Generate hypothetical answer for better retrieval
                        hyde_prompt = [
                            {
                                "role": "system",
                                "content": "Write a concise, factual paragraph that might answer the user's question."
                            },
                            {"role": "user", "content": query}
                        ]
                        hyde_answer = llm.invoke(hyde_prompt)
                        hyde_text = (
                            hyde_answer.get("content")
                            if isinstance(hyde_answer, dict)
                            else str(hyde_answer)
                        )
                        retrieval_query = query + "\n\n" + hyde_text
                    except Exception as e:
                        logger.warning(f"HyDE expansion failed: {e}")
                        retrieval_query = query
                
                # Retrieve documents
                t_retr_start = time.perf_counter()
                docs = retriever.retrieve(
                    retrieval_query,
                    k=settings["top_k"],
                    use_reranker=settings["use_reranker"],
                    rerank_top_k=settings["rerank_top_k"],
                    use_mmr=settings["use_mmr"],
                )
                
                # Apply optional filters
                filename_substring = settings.get("filename_filter", "").strip().lower()
                filetype_selection = set(settings.get("filetype_filter", []))
                
                if filename_substring or filetype_selection:
                    filtered_docs = []
                    for doc in docs:
                        # Check filename filter
                        source = (doc.metadata or {}).get("source", "")
                        filename_match = (
                            filename_substring in source.lower()
                            if filename_substring
                            else True
                        )
                        
                        # Check filetype filter
                        file_ext = (doc.metadata or {}).get("file_type") or (
                            Path(source).suffix if source else ""
                        )
                        filetype_match = (
                            file_ext in filetype_selection
                            if filetype_selection
                            else True
                        )
                        
                        if filename_match and filetype_match:
                            filtered_docs.append(doc)
                    
                    docs = filtered_docs or docs  # Fallback to unfiltered if no matches
                
                t_retr_end = time.perf_counter()
                
                # Handle empty retrieval
                if not docs:
                    msg = (
                        "I couldn't find relevant context. Try uploading more documents "
                        "or broadening your query."
                    )
                    st.markdown(msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": msg,
                        "sources": []
                    })
                    return
                
                # Format context for prompt
                def format_document(doc, idx):
                    """Format a document for inclusion in context."""
                    source = (doc.metadata or {}).get("source", "unknown")
                    content = doc.page_content
                    # Limit preview length for context
                    max_length = 1000
                    if len(content) > max_length:
                        content = content[:max_length] + "…"
                    return f"[{idx}] (source: {source})\n{content}\n"
                
                context = "\n".join([
                    format_document(doc, i + 1)
                    for i, doc in enumerate(docs)
                ])
                
                # Generate answer
                messages = prompt.format_messages(question=query, context=context)
                
                t_llm_start = time.perf_counter()
                answer = None
                deepconf_meta = None
                
                if settings.get("enable_deepconf"):
                    # Use DeepConf for confidence-aware generation
                    try:
                        # Convert messages to prompt text
                        prompt_text = ""
                        for msg in messages:
                            role = getattr(msg, "type", None) or getattr(msg, "role", "")
                            content = getattr(msg, "content", "")
                            prompt_text += (
                                (role.upper() + ": " if role else "") +
                                content + "\n\n"
                            )
                        
                        # Configure DeepConf
                        keep_ratio = 0.9 if settings.get("deepconf_keep") == "90%" else 0.1
                        dc = DeepConf(
                            Config.OLLAMA_HOST,
                            settings["model_name"],
                            topk=int(settings.get("deepconf_topk", 5)),
                            group_size=int(settings.get("deepconf_groupsize", 128)),
                        )
                        
                        # Run DeepConf decision
                        if settings.get("deepconf_mode") == "Offline":
                            res = dc.offline_decide(
                                prompt=prompt_text,
                                K=int(settings.get("deepconf_budget", 32)),
                                metric=settings.get("deepconf_metric", "bottom10"),
                                keep_ratio=keep_ratio,
                                temperature=settings["temperature"]
                            )
                        else:  # Online mode
                            mode = "high" if settings.get("deepconf_keep") == "90%" else "low"
                            res = dc.online_decide(
                                prompt=prompt_text,
                                budget=int(settings.get("deepconf_budget", 32)),
                                Ninit=int(settings.get("deepconf_ninit", 8)),
                                mode=mode,
                                metric=settings.get("deepconf_metric", "lowest_group"),
                                consensus_tau=float(settings.get("deepconf_tau", 0.95)),
                                temperature=settings["temperature"]
                            )
                        
                        answer = res.get("answer") or ""
                        deepconf_meta = res
                        
                    except Exception as e:
                        logger.error(f"DeepConf failed, falling back: {e}")
                        deepconf_meta = {"error": str(e)}
                        answer = llm.invoke(messages)
                else:
                    # Standard generation
                    answer = llm.invoke(messages)
                
                t_llm_end = time.perf_counter()
                
                # Prepare sources for display
                def create_preview(text: str, limit: int = 700) -> str:
                    """Create a preview of text with ellipsis if needed."""
                    return text if len(text) <= limit else (text[:limit] + "…")
                
                sources = [
                    {
                        "id": i + 1,
                        "source": doc.metadata.get("source", "Unknown"),
                        "preview": create_preview(doc.page_content),
                        "metadata": doc.metadata,
                    }
                    for i, doc in enumerate(docs)
                ]
                
                # Display answer
                st.markdown(answer)
                
                # Save to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "metrics": {
                        "retrieval_ms": int((t_retr_end - t_retr_start) * 1000),
                        "llm_ms": int((t_llm_end - t_llm_start) * 1000)
                    },
                    "deepconf": deepconf_meta
                })
                
                # Display sources
                if sources:
                    with st.expander("📚 Sources", expanded=False):
                        for src in sources:
                            st.markdown(f"**[{src['id']}]** {src['source']}")
                            st.code(src["preview"], language="text")
                
            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                logger.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })


# ====================================================================
# MAIN APPLICATION
# ====================================================================
def main():
    """
    Main application entry point.
    
    This function sets up the Streamlit app configuration,
    initializes session state, and renders the UI components.
    """
    # Page configuration
    st.set_page_config(
        page_title="Local RAG System",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Title and description
    st.title("🧠 Local RAG System")
    st.caption(
        "Enhanced RAG with Ollama, Chroma, and Streamlit • "
        "GPU-accelerated • WSL-optimized"
    )
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Create tabs for different functionality
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📁 Upload Documents", "ℹ️ Help"])
    
    with tab1:
        render_chat_interface(settings)
    
    with tab2:
        render_document_upload(settings)
    
    with tab3:
        # Help documentation
        st.markdown(
            f"""
            ### 🚀 Quick Start Guide
            
            1. **Ensure Ollama is running in WSL** with required models:
               - LLM: `llama3.1:8b` (or your preferred model)
               - Embeddings: `nomic-embed-text`
               
            2. **Upload documents** in the Upload tab:
               - Supported formats: PDF, TXT, MD, CSV, JSON
               - Maximum file size: 50MB
               
            3. **Ask questions** in the Chat tab:
               - Questions are answered based on your uploaded documents
               - Citations are provided for transparency
            
            ### 🔧 Troubleshooting
            
            **Cannot connect to Ollama:**
            - Verify Ollama is running: `ollama serve`
            - Check endpoint: `curl ${{OLLAMA_HOST}}/api/tags`
            - Ensure OLLAMA_HOST environment variable is set correctly
            
            **Slow performance:**
            - Reduce "Documents to retrieve" in settings
            - Decrease chunk size for faster processing
            - Disable reranker if not needed
            - Consider using GPU acceleration (install torch with CUDA)
            
            **Low answer quality:**
            - Enable reranker for better precision
            - Use Analytical prompt style for detailed answers
            - Enable DeepConf for confidence-aware reasoning
            - Increase "Documents to retrieve" for more context
            
            ### 📊 Configuration Guide
            
            **Retrieval Settings:**
            - **MMR (Maximum Marginal Relevance)**: Balances relevance with diversity
            - **Reranker**: Uses cross-encoder for precise ranking (GPU-accelerated)
            - **HyDE**: Generates hypothetical answer to improve retrieval
            
            **Prompt Styles:**
            - **Basic**: Concise, factual answers
            - **Analytical**: Detailed analysis with source comparison
            - **Creative**: Engaging, narrative explanations
            
            **DeepConf Settings:**
            - **Offline Mode**: Generate all traces upfront, then select best
            - **Online Mode**: Adaptive generation with early stopping
            - **Confidence Metrics**:
              - `bottom10`: Focus on worst-performing segments
              - `lowest_group`: Single worst sliding window
              - `tail`: Last N tokens (recency bias)
              - `avg`: Overall average confidence
            
            ### 🎯 Best Practices
            
            **Document Processing:**
            - Chunk size 400-800 for technical documents
            - Chunk size 800-1200 for narrative content
            - Higher overlap (100-200) for better context preservation
            
            **For Exploration:**
            - Enable MMR for diverse perspectives
            - Use higher "Documents to retrieve" (15-20)
            - Try Creative prompt style
            
            **For Precision:**
            - Enable reranker with GPU if available
            - Use lower "Documents to retrieve" (5-8)
            - Use Analytical prompt style
            - Enable DeepConf with high keep ratio (90%)
            
            ### 🔒 Privacy & Security
            
            - All processing happens locally (no external API calls)
            - Documents are stored in local Chroma database
            - No data leaves your WSL environment
            - Vector database persists between sessions
            
            ### 📈 Performance Tips
            
            1. **GPU Acceleration**: Install PyTorch with CUDA support for faster reranking
            2. **Model Selection**: Smaller models (7B-8B) offer good balance
            3. **Batch Processing**: Upload multiple documents at once
            4. **Cache Management**: Streamlit caches embeddings and models automatically
            
            ### 🐛 Debug Information
            
            - Logs are displayed in the terminal running Streamlit
            - Check Ollama logs: `journalctl -u ollama` (if using systemd)
            - Vector DB location: `{settings.get('db_dir', Config.DEFAULT_DB_DIR)}`
            - Current models: {settings.get('model_name', 'Not set')} / {settings.get('embed_name', 'Not set')}
            """
        )


# ====================================================================
# APPLICATION ENTRY POINT
# ====================================================================
if __name__ == "__main__":
    """
    Script entry point.
    
    This ensures the application only runs when executed directly,
    not when imported as a module.
    """
    main()