"""
RAG Chatbot - LLM interaction with context-aware answer generation

Flow:
    Context + Question → Prompt → Ollama/LLaMA → Answer
"""

from typing import List, Dict, Optional
from rich.console import Console
from rich.markdown import Markdown
import ollama

console = Console()


class RAGChatbot:
    """
    LLM-powered chatbot that answers questions using retrieved context.
    
    Features:
    - Context-aware prompting (RAG)
    - Ollama integration (local LLM)
    - Prevents hallucination (answers only from context)
    - Streaming responses
    """
    
    def __init__(
        self,
        model: str = "phi3",  # Ollama model name
        temperature: float = 0.1,  # Low = more focused, high = more creative
        max_tokens: int = 500  # Maximum response length
    ):
        """
        Initialize the RAG chatbot.
        
        Args:
            model: Ollama model name (e.g., 'llama3', 'mistral', 'phi')
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Check if Ollama is available
        self._check_ollama()
    
    def _check_ollama(self):
        """Check if Ollama is running and model is available."""
        try:
            # List available models
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]
            
            if self.model not in model_names and f"{self.model}:latest" not in model_names:
                console.print(f"[yellow]⚠️  Model '{self.model}' not found[/yellow]")
                console.print(f"[yellow]Available models: {', '.join(model_names)}[/yellow]")
                console.print(f"\n[cyan]To install: ollama pull {self.model}[/cyan]\n")
            else:
                console.print(f"[green]✓ Ollama model '{self.model}' ready[/green]")
                
        except Exception as e:
            console.print(f"[red]✗ Ollama not available: {str(e)}[/red]")
            console.print("\n[yellow]Make sure Ollama is running:[/yellow]")
            console.print("  1. Install from: https://ollama.com")
            console.print("  2. Start Ollama service")
            console.print(f"  3. Pull model: ollama pull {self.model}\n")
    
    def create_prompt(
        self,
        question: str,
        context: str,
        system_message: Optional[str] = None
    ) -> str:
        """
        Create a RAG prompt with context and question.
        
        Args:
            question: User's question
            context: Retrieved document chunks
            system_message: Optional system instructions
            
        Returns:
            Formatted prompt string
        """
        if system_message is None:
            system_message = (
                "You are a helpful AI assistant that answers questions based on provided context. "
                "IMPORTANT: Only use information from the context below. "
                "If the answer is not in the context, say 'I cannot answer this based on the provided documents.' "
                "Be concise, accurate, and cite sources when possible."
            )
        
        prompt = f"""{system_message}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        return prompt
    
    def generate_answer(
        self,
        question: str,
        context: str,
        stream: bool = False
    ) -> str:
        """
        Generate an answer using the LLM.
        
        Args:
            question: User's question
            context: Retrieved context
            stream: Whether to stream response
            
        Returns:
            Generated answer
        """
        # Create the prompt
        prompt = self.create_prompt(question, context)
        
        try:
            if stream:
                return self._generate_streaming(prompt)
            else:
                return self._generate_standard(prompt)
                
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg
    
    def _generate_standard(self, prompt: str) -> str:
        """Generate answer without streaming."""
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': self.temperature,
                'num_predict': self.max_tokens
            }
        )
        
        return response['response'].strip()
    
    def _generate_streaming(self, prompt: str) -> str:
        """Generate answer with streaming output."""
        console.print("\n[bold cyan]💬 Answer:[/bold cyan]\n")
        
        full_response = ""
        
        stream = ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=True,
            options={
                'temperature': self.temperature,
                'num_predict': self.max_tokens
            }
        )
        
        for chunk in stream:
            text = chunk['response']
            full_response += text
            console.print(text, end='')
        
        console.print("\n")
        
        return full_response
    
    def chat(
        self,
        question: str,
        retrieved_results: List[Dict[str, any]],
        stream: bool = True,
        show_context: bool = False
    ) -> Dict[str, str]:
        """
        Complete RAG chat interaction.
        
        Args:
            question: User's question
            retrieved_results: Results from VectorRetriever
            stream: Stream response
            show_context: Display retrieved context
            
        Returns:
            Dict with 'question', 'context', 'answer', 'sources'
        """
        if not retrieved_results:
            no_context_answer = (
                "I cannot answer this question as no relevant information "
                "was found in the indexed documents."
            )
            return {
                'question': question,
                'context': '',
                'answer': no_context_answer,
                'sources': []
            }
        
        # Format context from retrieved results
        context_parts = []
        sources = []
        
        for i, result in enumerate(retrieved_results, 1):
            context_parts.append(f"[Document {i}]\n{result['text']}")
            sources.append({
                'source': result['metadata']['source'],
                'score': result['score']
            })
        
        context = "\n\n".join(context_parts)
        
        # Show context if requested
        if show_context:
            console.print("\n[bold yellow]📄 Retrieved Context:[/bold yellow]")
            for i, result in enumerate(retrieved_results, 1):
                console.print(f"\n[cyan]Chunk {i} - {result['metadata']['source']} "
                            f"(score: {result['score']:.2%})[/cyan]")
                console.print(result['text'][:300] + "...")
            console.print("\n" + "="*80 + "\n")
        
        # Generate answer
        answer = self.generate_answer(question, context, stream=stream)
        
        return {
            'question': question,
            'context': context,
            'answer': answer,
            'sources': sources
        }
    
    def display_result(self, result: Dict[str, str]):
        """
        Display chat result in a formatted way.
        
        Args:
            result: Result from chat()
        """
        console.print("\n[bold green]📝 Answer:[/bold green]")
        console.print(Markdown(result['answer']))
        
        if result['sources']:
            console.print("\n[bold cyan]📚 Sources:[/bold cyan]")
            for source in result['sources']:
                console.print(f"  • {source['source']} (relevance: {source['score']:.2%})")


def main():
    """Test the chatbot with a sample question."""
    from retrieve import VectorRetriever
    
    console.print("\n[bold cyan]╔════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║   RAG CHATBOT TEST MODE       ║[/bold cyan]")
    console.print("[bold cyan]╚════════════════════════════════╝[/bold cyan]\n")
    
    # Initialize components
    retriever = VectorRetriever()
    chatbot = RAGChatbot()
    
    if retriever.index is None:
        console.print("[red]No index found. Run ingestion first.[/red]")
        return
    
    # Test question
    question = "What are the main topics discussed in the documents?"
    
    console.print(f"[bold yellow]❓ Question:[/bold yellow] {question}\n")
    
    # Retrieve context
    results = retriever.retrieve(question, top_k=3)
    
    # Generate answer
    result = chatbot.chat(question, results, stream=True, show_context=True)
    
    # Display
    chatbot.display_result(result)


if __name__ == "__main__":
    main()
