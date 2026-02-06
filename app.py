# MUST be set before any imports to avoid TensorFlow DLL issues
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

"""
DOCLING RAG - Complete Document Q&A System

A free, local RAG system for answering questions about PDFs with OCR support.

Architecture:
    PDFs → Docling (OCR) → Embeddings → FAISS → LLaMA → Answers
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest import PDFIngestor
from retrieve import VectorRetriever
from chat import RAGChatbot

console = Console()


class DoclingRAGApp:
    """Main application controller for the RAG system."""
    
    def __init__(self):
        """Initialize the RAG application."""
        self.ingestor = None
        self.retriever = None
        self.chatbot = None
        self.index_exists = False
    
    def check_system(self) -> bool:
        """
        Check if the system is ready to use.
        
        Returns:
            True if system is ready, False otherwise
        """
        console.print("\n[bold cyan]🔍 System Check[/bold cyan]\n")
        
        checks = []
        
        # Check 1: GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                checks.append(("GPU", f"✓ {gpu_name}", "green"))
            else:
                checks.append(("GPU", "✗ Not available (CPU mode)", "yellow"))
        except:
            checks.append(("GPU", "✗ PyTorch not installed", "yellow"))
        
        # Check 2: Index exists
        index_path = Path("index/faiss.index")
        if index_path.exists():
            checks.append(("Index", "✓", "green"))
            self.index_exists = True
        else:
            checks.append(("Index", "✗ Not found", "yellow"))
            self.index_exists = False
        
        # Check 3: PDFs exist
        pdf_folder = Path("data/pdfs")
        pdf_count = len(list(pdf_folder.glob("*.pdf"))) if pdf_folder.exists() else 0
        if pdf_count > 0:
            checks.append(("PDFs", f"✓ {pdf_count} found", "green"))
        else:
            checks.append(("PDFs", "✗ None found", "yellow"))
        
        # Check 4: Ollama and phi3 model
        try:
            import ollama
            result = ollama.list()
            models = result.get('models', []) if isinstance(result, dict) else result.models if hasattr(result, 'models') else []
            
            if not models:
                checks.append(("Ollama/phi3", "⚠️ No models found", "yellow"))
            else:
                # Extract model names (handle both dict and object formats)
                model_names = []
                for m in models:
                    if isinstance(m, dict):
                        model_names.append(m.get('name', m.get('model', '')))
                    else:
                        model_names.append(getattr(m, 'name', getattr(m, 'model', '')))
                
                # Check for phi3 model (any variant)
                phi3_found = any('phi3' in str(name).lower() for name in model_names if name)
                
                if phi3_found:
                    checks.append(("Ollama/phi3", "✓ Ready", "green"))
                else:
                    checks.append(("Ollama/phi3", f"✗ Model not found", "yellow"))
                    if model_names:
                        console.print(f"[dim]  Available: {', '.join(str(n) for n in model_names[:3])}[/dim]")
        except Exception as e:
            checks.append(("Ollama/phi3", f"✗ Error: {str(e)[:30]}", "red"))
        
        # Display checks
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="white")
        
        for name, status, color in checks:
            table.add_row(name, f"[{color}]{status}[/{color}]")
        
        console.print(table)
        console.print()
        
        return True
    
    def show_welcome(self):
        """Display welcome message."""
        welcome_text = """
[bold cyan]📚 DOCLING RAG SYSTEM[/bold cyan]
[dim]Local, Free Document Q&A with OCR Support[/dim]

[yellow]Features:[/yellow]
• OCR for scanned PDFs
• Semantic search (not just keywords)
• Local LLM (private, no API costs)
• Completely offline

[cyan]Stack:[/cyan] Docling + SentenceTransformers + FAISS + Ollama
        """
        
        panel = Panel(welcome_text, expand=False, border_style="cyan")
        console.print(panel)
    
    def run_ingestion(self):
        """Run the PDF ingestion pipeline."""
        console.print("\n[bold yellow]📥 INGESTION MODE[/bold yellow]\n")
        
        pdf_folder = Path("data/pdfs")
        pdf_count = len(list(pdf_folder.glob("*.pdf"))) if pdf_folder.exists() else 0
        
        if pdf_count == 0:
            console.print("[yellow]⚠️  No PDFs found in data/pdfs/[/yellow]")
            console.print("\n[cyan]Please add PDF files to data/pdfs/ and try again.[/cyan]\n")
            return False
        
        console.print(f"[green]Found {pdf_count} PDF(s) to process[/green]\n")
        
        if self.index_exists:
            overwrite = Confirm.ask(
                "[yellow]Index already exists. Overwrite?[/yellow]",
                default=False
            )
            if not overwrite:
                console.print("[cyan]Ingestion cancelled.[/cyan]\n")
                return False
        
        # Run ingestion
        self.ingestor = PDFIngestor()
        self.ingestor.run_pipeline()
        
        self.index_exists = True
        return True
    
    def run_chat(self):
        """Run the interactive chat interface."""
        if not self.index_exists:
            console.print("\n[red]❌ No index found![/red]")
            console.print("[yellow]Run ingestion first to process PDFs.[/yellow]\n")
            return
        
        console.print("\n[bold green]💬 CHAT MODE[/bold green]")
        console.print("[dim]Ask questions about your documents. Type 'quit' or 'exit' to stop.[/dim]\n")
        
        # Initialize components
        if self.retriever is None:
            self.retriever = VectorRetriever()
        
        if self.chatbot is None:
            self.chatbot = RAGChatbot()
        
        # Check if retriever loaded successfully
        if self.retriever.index is None:
            console.print("[red]Failed to load index. Cannot start chat.[/red]\n")
            return
        
        # Chat loop
        while True:
            try:
                # Get question from user
                question = Prompt.ask("\n[bold cyan]❓ Your question[/bold cyan]")
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'q']:
                    console.print("\n[cyan]👋 Goodbye![/cyan]\n")
                    break
                
                if not question.strip():
                    continue
                
                # Retrieve more chunks initially, let chatbot filter by relevance
                console.print("\n[dim]🔍 Searching documents...[/dim]")
                results = self.retriever.retrieve(question, top_k=7)
                
                if not results:
                    console.print("[yellow]⚠️  No relevant information found.[/yellow]")
                    continue
                
                # Show retrieval stats
                console.print(f"[dim]Found {len(results)} chunks (best: {results[0]['score']:.0%} relevance)[/dim]")
                
                # Generate answer
                result = self.chatbot.chat(
                    question,
                    results,
                    stream=True,
                    show_context=False
                )
                
                # Display the answer if not already streamed
                if not result.get('answer'):
                    console.print("[red]⚠️  No answer generated[/red]")
                
                # Show sources
                if result['sources']:
                    console.print("\n[bold dim]📚 Sources:[/bold dim]")
                    for source in result['sources']:
                        console.print(
                            f"[dim]  • {source['source']} "
                            f"(relevance: {source['score']:.0%})[/dim]"
                        )
                
            except KeyboardInterrupt:
                console.print("\n\n[cyan]👋 Goodbye![/cyan]\n")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
    
    def show_menu(self) -> str:
        """
        Show main menu and get user choice.
        
        Returns:
            User's choice
        """
        console.print("\n[bold cyan]What would you like to do?[/bold cyan]\n")
        
        options = []
        
        if not self.index_exists:
            options.append("1. [yellow]Ingest PDFs[/yellow] (required first step)")
        else:
            options.append("1. [yellow]Re-ingest PDFs[/yellow] (rebuild index)")
        
        if self.index_exists:
            options.append("2. [green]Chat with documents[/green]")
        else:
            options.append("2. [dim]Chat with documents[/dim] (requires ingestion)")
        
        options.append("3. [cyan]Check system status[/cyan]")
        options.append("4. [red]Exit[/red]")
        
        for option in options:
            console.print(f"  {option}")
        
        console.print()
        
        choice = Prompt.ask(
            "[bold]Choose an option[/bold]",
            choices=["1", "2", "3", "4"],
            default="2" if self.index_exists else "1"
        )
        
        return choice
    
    def run(self):
        """Run the main application loop."""
        self.show_welcome()
        self.check_system()
        
        while True:
            try:
                choice = self.show_menu()
                
                if choice == "1":
                    # Ingest PDFs
                    self.run_ingestion()
                
                elif choice == "2":
                    # Chat
                    self.run_chat()
                
                elif choice == "3":
                    # System check
                    self.check_system()
                
                elif choice == "4":
                    # Exit
                    console.print("\n[cyan]👋 Goodbye![/cyan]\n")
                    break
                
            except KeyboardInterrupt:
                console.print("\n\n[cyan]👋 Goodbye![/cyan]\n")
                break
            except Exception as e:
                console.print(f"\n[red]Unexpected error: {str(e)}[/red]")
                console.print("[yellow]Please report this issue.[/yellow]\n")


def main():
    """Entry point for the application."""
    app = DoclingRAGApp()
    app.run()


if __name__ == "__main__":
    main()
