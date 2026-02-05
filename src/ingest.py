"""
PDF Ingestion Pipeline - Processes PDFs into searchable vector database

Flow:
    PDF → Docling (OCR + extraction) → Text chunks → Embeddings → FAISS index
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Docling for PDF processing with OCR
from docling.document_converter import DocumentConverter

# SentenceTransformers for embeddings
from sentence_transformers import SentenceTransformer

# FAISS for vector storage and search
import faiss

console = Console()


class PDFIngestor:
    """
    Processes PDFs and builds a searchable vector database.
    
    Features:
    - Automatic OCR for scanned documents
    - Hierarchical chunking (preserves structure)
    - Semantic embeddings for search
    - FAISS index for fast retrieval
    """
    
    def __init__(
        self,
        pdf_folder: str = "data/pdfs",
        index_folder: str = "index",
        embedding_model: str = "all-MiniLM-L6-v2",  # Fast, 384-dim embeddings
        chunk_size: int = 500  # Characters per chunk
    ):
        """
        Initialize the PDF ingestor.
        
        Args:
            pdf_folder: Folder containing PDFs to process
            index_folder: Where to save FAISS index and metadata
            embedding_model: SentenceTransformer model name
            chunk_size: Approximate characters per text chunk
        """
        self.pdf_folder = Path(pdf_folder)
        self.index_folder = Path(index_folder)
        self.chunk_size = chunk_size
        
        # Create folders if they don't exist
        self.pdf_folder.mkdir(parents=True, exist_ok=True)
        self.index_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        console.print(f"[cyan]Loading embedding model: {embedding_model}...[/cyan]")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Docling converter (handles OCR automatically)
        self.converter = DocumentConverter()
        
        # Storage for chunks and metadata
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []
        
    def process_pdfs(self) -> Tuple[int, int]:
        """
        Process all PDFs in the folder.
        
        Returns:
            (num_pdfs, num_chunks): Number of PDFs and chunks processed
        """
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        
        if not pdf_files:
            console.print("[yellow]⚠️  No PDFs found in data/pdfs/[/yellow]")
            return 0, 0
        
        console.print(f"[green]Found {len(pdf_files)} PDF(s) to process[/green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for pdf_path in pdf_files:
                task = progress.add_task(
                    f"Processing {pdf_path.name}...",
                    total=None
                )
                
                try:
                    # Docling automatically detects and runs OCR if needed
                    result = self.converter.convert(str(pdf_path))
                    
                    # Extract text from document
                    text = result.document.export_to_markdown()
                    
                    if not text.strip():
                        console.print(f"[yellow]⚠️  {pdf_path.name}: No text extracted[/yellow]")
                        continue
                    
                    # Split into chunks
                    chunks = self._chunk_text(text)
                    
                    # Store chunks with metadata
                    for i, chunk in enumerate(chunks):
                        self.chunks.append(chunk)
                        self.metadata.append({
                            "source": pdf_path.name,
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        })
                    
                    console.print(
                        f"[green]✓ {pdf_path.name}: {len(chunks)} chunks created[/green]"
                    )
                    
                except Exception as e:
                    console.print(f"[red]✗ {pdf_path.name}: {str(e)}[/red]")
                
                progress.remove_task(task)
        
        return len(pdf_files), len(self.chunks)
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap for better context.
        
        Args:
            text: Full document text
            
        Returns:
            List of text chunks
        """
        chunks = []
        overlap = int(self.chunk_size * 0.2)  # 20% overlap
        
        # Split into paragraphs first (preserves structure)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        
        for para in paragraphs:
            # If paragraph alone is too big, split it
            if len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Split large paragraph
                words = para.split()
                for word in words:
                    if len(current_chunk) + len(word) > self.chunk_size:
                        chunks.append(current_chunk)
                        # Keep overlap
                        current_chunk = ' '.join(current_chunk.split()[-overlap:]) + ' ' + word
                    else:
                        current_chunk += ' ' + word if current_chunk else word
            else:
                # Add paragraph to chunk
                if len(current_chunk) + len(para) > self.chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = para
                else:
                    current_chunk += '\n\n' + para if current_chunk else para
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def create_embeddings(self) -> np.ndarray:
        """
        Convert text chunks to vector embeddings.
        
        Returns:
            numpy array of embeddings (shape: num_chunks × embedding_dim)
        """
        if not self.chunks:
            console.print("[yellow]⚠️  No chunks to embed[/yellow]")
            return np.array([])
        
        console.print(f"[cyan]Creating embeddings for {len(self.chunks)} chunks...[/cyan]")
        
        # Convert chunks to vectors (this is where the magic happens!)
        embeddings = self.embedder.encode(
            self.chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        console.print(
            f"[green]✓ Embeddings created: {embeddings.shape[0]} × {embeddings.shape[1]}D[/green]"
        )
        
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build FAISS index for fast similarity search.
        
        Args:
            embeddings: numpy array of embeddings
            
        Returns:
            FAISS index
        """
        if embeddings.size == 0:
            console.print("[yellow]⚠️  No embeddings to index[/yellow]")
            return None
        
        console.print("[cyan]Building FAISS index...[/cyan]")
        
        # Get embedding dimension
        dimension = embeddings.shape[1]
        
        # Create FAISS index (L2 distance for similarity)
        # IndexFlatL2 = brute force, exact search (good for small datasets)
        index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to index
        index.add(embeddings.astype('float32'))
        
        console.print(
            f"[green]✓ FAISS index built: {index.ntotal} vectors indexed[/green]"
        )
        
        return index
    
    def save_index(self, index: faiss.Index):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index: FAISS index to save
        """
        if index is None:
            return
        
        # Save FAISS index
        index_path = self.index_folder / "faiss.index"
        faiss.write_index(index, str(index_path))
        
        # Save chunks and metadata
        data_path = self.index_folder / "chunks.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata
            }, f)
        
        console.print(f"[green]✓ Index saved to {self.index_folder}/[/green]")
    
    def run_pipeline(self):
        """
        Run the complete ingestion pipeline.
        
        Steps:
            1. Process PDFs (extract text, OCR if needed)
            2. Create embeddings (text → vectors)
            3. Build FAISS index (vector database)
            4. Save to disk
        """
        console.print("\n[bold cyan]╔═══════════════════════════════════╗[/bold cyan]")
        console.print("[bold cyan]║   PDF INGESTION PIPELINE START   ║[/bold cyan]")
        console.print("[bold cyan]╚═══════════════════════════════════╝[/bold cyan]\n")
        
        # Step 1: Process PDFs
        num_pdfs, num_chunks = self.process_pdfs()
        
        if num_chunks == 0:
            console.print("\n[red]Pipeline stopped: No chunks created[/red]")
            return
        
        # Step 2: Create embeddings
        embeddings = self.create_embeddings()
        
        # Step 3: Build FAISS index
        index = self.build_faiss_index(embeddings)
        
        # Step 4: Save everything
        self.save_index(index)
        
        console.print("\n[bold green]╔═══════════════════════════════════╗[/bold green]")
        console.print("[bold green]║   PIPELINE COMPLETED SUCCESSFULLY  ║[/bold green]")
        console.print("[bold green]╚═══════════════════════════════════╝[/bold green]")
        console.print(f"\n[cyan]📊 Summary:[/cyan]")
        console.print(f"  • PDFs processed: {num_pdfs}")
        console.print(f"  • Chunks created: {num_chunks}")
        console.print(f"  • Index location: {self.index_folder}/")
        console.print(f"\n[green]Ready for Q&A! 🚀[/green]\n")


def main():
    """Run the ingestion pipeline."""
    ingestor = PDFIngestor()
    ingestor.run_pipeline()


if __name__ == "__main__":
    main()
