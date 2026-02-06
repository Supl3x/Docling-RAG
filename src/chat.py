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
        max_tokens: int = 1000,  # Maximum response length
        min_relevance_score: float = 0.35  # Balanced threshold for quality
    ):
        """
        Initialize the RAG chatbot.
        
        Args:
            model: Ollama model name (e.g., 'llama3', 'mistral', 'phi')
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            min_relevance_score: Minimum score to include a chunk in context
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_relevance_score = min_relevance_score
        
        # Check if Ollama is available
        self._check_ollama()
    
    def _check_ollama(self):
        """Check if Ollama is running and model is available."""
        try:
            # List available models
            result = ollama.list()
            models = result.get('models', []) if isinstance(result, dict) else result.models if hasattr(result, 'models') else []
            
            # Extract model names
            model_names = []
            for m in models:
                if isinstance(m, dict):
                    model_names.append(m.get('name', m.get('model', '')))
                else:
                    model_names.append(getattr(m, 'name', getattr(m, 'model', '')))
            
            # Check if our model exists (with or without :latest tag)
            model_found = False
            for name in model_names:
                if name and (name == self.model or name == f"{self.model}:latest" or name.startswith(f"{self.model}:")):
                    model_found = True
                    break
            
            if not model_found:
                console.print(f"[yellow]⚠️  Model '{self.model}' not found[/yellow]")
                console.print(f"[yellow]Available models: {', '.join(str(n) for n in model_names if n)}[/yellow]")
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
                "You are an AI assistant that answers questions using ONLY the provided context.\n\n"
                "STRICT RULES:\n"
                "1. Answer ONLY using information from the CONTEXT section below\n"
                "2. If the context contains tables, marks, numbers, or data - quote them EXACTLY as shown\n"
                "3. If asked about specific names, numbers, or values - look for EXACT matches in the context\n"
                "4. Do NOT make up, infer, or add any information not present in the context\n"
                "5. If the answer is not in the context, say: 'This information is not found in the document.'\n"
                "6. When presenting data from tables, format it clearly and cite the source\n"
                "7. Be concise and direct - do not add unnecessary explanations"
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
        # Build chat messages with proper system/user roles for better comprehension
        system_message = (
            "You are an AI assistant that answers questions using ONLY the provided context.\n\n"
            "STRICT RULES:\n"
            "1. Answer ONLY using information from the CONTEXT provided by the user\n"
            "2. If the context contains tables, numbers, or data - quote them EXACTLY\n"
            "3. If asked about specific names, numbers, or values - look for EXACT matches\n"
            "4. Do NOT make up or infer information not in the context\n"
            "5. If the answer is not in the context, say: 'This information is not found in the document.'\n"
            "6. Be concise and direct"
        )
        
        user_content = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nProvide a direct answer based only on the context above."
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        try:
            if stream:
                return self._generate_streaming(messages)
            else:
                return self._generate_standard(messages)
                
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg
    
    def _generate_standard(self, messages: list) -> str:
        """Generate answer without streaming using chat API."""
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                'temperature': self.temperature,
                'num_predict': self.max_tokens,
                'num_gpu': 15,  # Limit GPU layers for 2GB VRAM
                'num_thread': 4
            }
        )
        
        return response['message']['content'].strip()
    
    def _generate_streaming(self, messages: list) -> str:
        """Generate answer with streaming output using chat API."""
        console.print("\n[bold cyan]💬 Answer:[/bold cyan]\n")
        
        full_response = ""
        
        stream = ollama.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={
                'temperature': self.temperature,
                'num_predict': self.max_tokens,
                'num_gpu': 15,  # Limit GPU layers for 2GB VRAM
                'num_thread': 4
            }
        )
        
        for chunk in stream:
            text = chunk['message']['content']
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
        
        # Filter by minimum relevance score for better quality
        filtered_results = [
            r for r in retrieved_results 
            if r['score'] >= self.min_relevance_score
        ]
        
        if not filtered_results:
            return {
                'question': question,
                'context': '',
                'answer': "No sufficiently relevant information was found in the documents for this question. Try rephrasing or asking about a different topic covered in your PDFs.",
                'sources': []
            }
        
        # Format context from filtered results (only top relevant chunks)
        context_parts = []
        sources = []
        
        # Limit to top 4 most relevant chunks to fit phi3 context window
        top_results = filtered_results[:4]
        
        for i, result in enumerate(top_results, 1):
            context_parts.append(f"[Document {i} - Relevance: {result['score']:.0%}]\n{result['text']}")
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
