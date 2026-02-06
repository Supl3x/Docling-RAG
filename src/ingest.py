# MUST be set before any imports to avoid TensorFlow DLL issues
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Fix SSL certificate issues on Windows
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

"""
PDF Ingestion Pipeline - Processes PDFs into searchable vector database

Flow:
    PDF → Docling (OCR + extraction) → Text chunks → Embeddings → FAISS index
"""

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
        embedding_model: str = "BAAI/bge-small-en-v1.5",  # Better retrieval quality, 384-dim, lightweight
        chunk_size: int = 1200,  # Balanced size for good context and precision
        chunk_overlap: int = 200  # Overlap to preserve context across chunks
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
        self.chunk_overlap = chunk_overlap
        
        # Create folders if they don't exist
        self.pdf_folder.mkdir(parents=True, exist_ok=True)
        self.index_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
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
        Split text into semantic chunks preserving tables and structure.
        
        Args:
            text: Full document text (markdown format)
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # First, identify and protect tables/structured data
        lines = text.split('\n')
        segments = []
        current_segment = []
        in_table = False
        
        for line in lines:
            # Detect table rows (markdown tables have | characters)
            is_table_line = '|' in line and line.strip().startswith('|')
            
            if is_table_line and not in_table:
                # Starting a table - save current segment first
                if current_segment:
                    segments.append(('text', '\n'.join(current_segment)))
                    current_segment = []
                in_table = True
                current_segment = [line]
            elif is_table_line and in_table:
                # Continue table
                current_segment.append(line)
            elif not is_table_line and in_table:
                # End of table
                segments.append(('table', '\n'.join(current_segment)))
                current_segment = [line] if line.strip() else []
                in_table = False
            else:
                # Regular text
                current_segment.append(line)
        
        # Don't forget last segment
        if current_segment:
            segment_type = 'table' if in_table else 'text'
            segments.append((segment_type, '\n'.join(current_segment)))
        
        # Now chunk each segment appropriately
        for seg_type, content in segments:
            if not content.strip():
                continue
                
            if seg_type == 'table':
                # Keep tables intact - don't split them
                # But if table is huge, split by rows
                if len(content) > self.chunk_size * 2:
                    table_lines = content.split('\n')
                    header = table_lines[:2] if len(table_lines) > 2 else table_lines[:1]
                    header_text = '\n'.join(header)
                    
                    current_table_chunk = header_text
                    for row in table_lines[len(header):]:
                        if len(current_table_chunk) + len(row) > self.chunk_size:
                            chunks.append(current_table_chunk)
                            current_table_chunk = header_text + '\n' + row
                        else:
                            current_table_chunk += '\n' + row
                    if current_table_chunk:
                        chunks.append(current_table_chunk)
                else:
                    chunks.append(content)
            else:
                # Regular text - chunk with overlap
                self._chunk_regular_text(content, chunks)
        
        return chunks
    
    def _chunk_regular_text(self, text: str, chunks: List[str]):
        """
        Chunk regular text with sentence-aware boundaries and proper overlap.
        
        Uses self.chunk_overlap to carry context between chunks.
        
        Args:
            text: Text to chunk
            chunks: List to append chunks to
        """
        import re
        
        # Split into sentences - require capital letter after punctuation
        # to avoid splitting on abbreviations like "Dr.", "U.S.", "3.14"
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        current_chunk = ""
        previous_sentences = []  # Track sentences for overlap
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) + 1 > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Build overlap from previous sentences up to chunk_overlap chars
                overlap = ""
                for prev in reversed(previous_sentences):
                    candidate = prev + " " + overlap if overlap else prev
                    if len(candidate) <= self.chunk_overlap:
                        overlap = candidate
                    else:
                        break
                
                current_chunk = overlap + " " + sentence if overlap else sentence
                previous_sentences = [sentence]
            elif not current_chunk and len(sentence) > self.chunk_size:
                # Single sentence too long - add it as-is
                chunks.append(sentence)
                previous_sentences = []
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                previous_sentences.append(sentence)
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    def create_embeddings(self) -> np.ndarray:
        """
        Convert text chunks to vector embeddings.
        
        Returns:
            numpy array of embeddings (shape: num_chunks × embedding_dim)
        """
        if not self.chunks:
            console.print("[yellow]⚠️  No chunks to embed[/yellow]")
            return np.array([])
        
        console.print(f"[cyan]Creating embeddings for {len(self.chunks)} chunks on {self.device.upper()}...[/cyan]")
        
        # Convert chunks to vectors with GPU optimization
        # Use batch_size for better GPU utilization
        batch_size = 32 if self.device == 'cuda' else 16  # Reduced for 2GB VRAM
        
        embeddings = self.embedder.encode(
            self.chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=batch_size,
            normalize_embeddings=True  # Normalize for cosine similarity
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
        
        # Use Inner Product (cosine similarity on normalized vectors) for better accuracy
        # IndexFlatIP = brute force, exact search with inner product
        index = faiss.IndexFlatIP(dimension)
        
        # Add vectors to index (already normalized)
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
