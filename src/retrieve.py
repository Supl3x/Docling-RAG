# MUST be set before any imports to avoid TensorFlow DLL issues
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

"""
Vector Retrieval Engine - Searches vector database for relevant content

Flow:
    Question → Embedding → FAISS search → Top-K similar chunks → Context
"""

import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from rich.console import Console

# SentenceTransformers for query embeddings
from sentence_transformers import SentenceTransformer

# FAISS for vector search
import faiss

console = Console()


class VectorRetriever:
    """
    Searches the vector database to find relevant document chunks.
    
    Features:
    - Loads pre-built FAISS index
    - Converts questions to embeddings
    - Performs semantic similarity search
    - Returns ranked results with scores
    """
    
    def __init__(
        self,
        index_folder: str = "index",
        embedding_model: str = "BAAI/bge-small-en-v1.5",  # Must match ingestion model
        top_k: int = 5  # Number of results to return
    ):
        """
        Initialize the vector retriever.
        
        Args:
            index_folder: Folder containing FAISS index and metadata
            embedding_model: SentenceTransformer model (must match ingest.py)
            top_k: Number of most relevant chunks to retrieve
        """
        self.index_folder = Path(index_folder)
        self.top_k = top_k
        
        # Load embedding model (same as used during ingestion)
        console.print(f"[cyan]Loading embedding model: {embedding_model}...[/cyan]")
        
        # Detect and use GPU if available
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            console.print(f"[green]✓ GPU detected: {torch.cuda.get_device_name(0)}[/green]")
            # Enable GPU optimizations
            torch.backends.cudnn.benchmark = True
        else:
            console.print("[yellow]⚠️  No GPU detected, using CPU[/yellow]")
        
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        
        # Load FAISS index and metadata
        self.index = None
        self.chunks = []
        self.metadata = []
        self._load_index()
    
    def _load_index(self):
        """Load FAISS index and chunk metadata from disk."""
        index_path = self.index_folder / "faiss.index"
        data_path = self.index_folder / "chunks.pkl"
        
        # Check if index exists
        if not index_path.exists():
            console.print(f"[yellow]⚠️  No index found at {index_path}[/yellow]")
            console.print("[yellow]Run ingestion first: python src/ingest.py[/yellow]")
            return
        
        if not data_path.exists():
            console.print(f"[yellow]⚠️  No chunk data found at {data_path}[/yellow]")
            return
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load chunks and metadata
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']
            
            console.print(
                f"[green]✓ Index loaded: {self.index.ntotal} vectors, "
                f"{len(self.chunks)} chunks[/green]"
            )
            
        except Exception as e:
            console.print(f"[red]Error loading index: {str(e)}[/red]")
    
    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> List[Dict[str, any]]:
        """
        Retrieve most relevant chunks for a query.
        
        Args:
            query: User's question or search query
            top_k: Number of results (overrides default)
            
        Returns:
            List of dicts with keys: 'text', 'score', 'metadata'
            Sorted by relevance (most relevant first)
        """
        if self.index is None:
            console.print("[red]Index not loaded. Cannot retrieve.[/red]")
            return []
        
        if not query.strip():
            console.print("[yellow]Empty query provided[/yellow]")
            return []
        
        k = top_k or self.top_k
        
        # Ensure we don't request more than available
        k = min(k, self.index.ntotal)
        
        try:
            # Convert query to embedding with GPU optimization
            # Add instruction prefix for BGE model (improves retrieval accuracy)
            query_text = f"Represent this sentence for searching relevant passages: {query}"
            
            # Convert query to embedding with GPU optimization
            # Normalize for cosine similarity (matching how chunks were embedded)
            query_embedding = self.embedder.encode(
                [query_text],
                convert_to_numpy=True,
                normalize_embeddings=True  # Must match ingestion
            )
            
            # Search FAISS index for similar vectors
            # scores: Inner product scores (higher = more similar)
            # indices: positions in the index
            scores, indices = self.index.search(
                query_embedding.astype('float32'),
                k
            )
            
            # Build results
            results = []
            for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
                # Inner product on normalized vectors = cosine similarity (0 to 1)
                # Clamp to valid range
                similarity_score = max(0.0, min(1.0, float(score)))
                
                results.append({
                    'text': self.chunks[idx],
                    'score': similarity_score,
                    'distance': 1.0 - similarity_score,  # For backward compatibility
                    'metadata': self.metadata[idx],
                    'rank': i + 1
                })
            
            return results
            
        except Exception as e:
            console.print(f"[red]Error during retrieval: {str(e)}[/red]")
            return []
    
    def retrieve_with_threshold(
        self,
        query: str,
        similarity_threshold: float = 0.5,
        max_results: int = 10
    ) -> List[Dict[str, any]]:
        """
        Retrieve chunks above a similarity threshold.
        
        Args:
            query: User's question
            similarity_threshold: Minimum similarity score (0-1)
            max_results: Maximum number of results
            
        Returns:
            List of relevant chunks above threshold
        """
        # Get more results than needed
        all_results = self.retrieve(query, top_k=max_results)
        
        # Filter by threshold
        filtered = [
            r for r in all_results 
            if r['score'] >= similarity_threshold
        ]
        
        return filtered
    
    def format_context(
        self,
        results: List[Dict[str, any]],
        include_scores: bool = False,
        include_sources: bool = True,
        min_score: float = 0.35  # Filter out low relevance chunks
    ) -> str:
        """
        Format retrieved chunks into context string for LLM.
        
        Args:
            results: Retrieved chunks from retrieve()
            include_scores: Show relevance scores
            include_sources: Show source document names
            min_score: Minimum similarity score to include
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        # Filter by minimum score for quality
        filtered_results = [r for r in results if r['score'] >= min_score]
        
        if not filtered_results:
            return "No sufficiently relevant information found."
        
        context_parts = []
        
        for result in filtered_results:
            # Build chunk header
            header_parts = []
            if include_sources:
                source = result['metadata']['source']
                header_parts.append(f"Source: {source}")
            if include_scores:
                score = result['score']
                header_parts.append(f"Relevance: {score:.2%}")
            
            if header_parts:
                header = " | ".join(header_parts)
                context_parts.append(f"[{header}]")
            
            # Add the actual text
            context_parts.append(result['text'])
            context_parts.append("")  # Blank line between chunks
        
        return "\n".join(context_parts)
    
    def search_and_display(self, query: str, top_k: int = None):
        """
        Retrieve and display results (for testing/debugging).
        
        Args:
            query: Search query
            top_k: Number of results
        """
        console.print(f"\n[bold cyan]🔍 Query:[/bold cyan] {query}\n")
        
        results = self.retrieve(query, top_k)
        
        if not results:
            console.print("[yellow]No results found[/yellow]")
            return
        
        console.print(f"[green]Found {len(results)} relevant chunks:[/green]\n")
        
        for result in results:
            console.print(f"[bold]Rank #{result['rank']}[/bold]")
            console.print(f"Source: {result['metadata']['source']}")
            console.print(f"Score: {result['score']:.2%} (distance: {result['distance']:.4f})")
            console.print(f"\n{result['text'][:200]}...\n")
            console.print("-" * 80)


def main():
    """Test the retriever with sample queries."""
    retriever = VectorRetriever()
    
    if retriever.index is None:
        console.print("\n[red]Cannot test: No index found[/red]")
        console.print("[yellow]Run ingestion first: python src/ingest.py[/yellow]")
        return
    
    # Sample queries for testing
    test_queries = [
        "What are the main topics?",
        "Tell me about the methodology",
        "What are the conclusions?"
    ]
    
    for query in test_queries:
        retriever.search_and_display(query, top_k=3)
        console.print("\n")


if __name__ == "__main__":
    main()
