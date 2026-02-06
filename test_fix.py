"""Quick test to verify the answer generation fix"""

import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieve import VectorRetriever
from chat import RAGChatbot

print("=" * 60)
print("TESTING ANSWER GENERATION FIX")
print("=" * 60)

# Initialize components
print("\nInitializing retriever...")
retriever = VectorRetriever()

print("\nInitializing chatbot...")
chatbot = RAGChatbot()

if retriever.index is None:
    print("\n❌ No index found! Run ingestion first.")
    sys.exit(1)

# Test question
question = "What is Feras Malik's roll number?"

print(f"\n📝 Question: {question}")
print("\n" + "=" * 60)

# Retrieve
print("\n🔍 Retrieving relevant chunks...")
results = retriever.retrieve(question, top_k=10)

print(f"\n✓ Found {len(results)} chunks")
for i, r in enumerate(results[:5], 1):
    print(f"  {i}. Score: {r['score']:.2%} | {r['metadata']['source']}")
    print(f"     Preview: {r['text'][:100]}...")

# Generate answer
print("\n💬 Generating answer...")
print("=" * 60)

result = chatbot.chat(
    question,
    results,
    stream=True,
    show_context=False
)

print("\n" + "=" * 60)
print(f"✓ Answer generated: {len(result['answer'])} characters")
print(f"✓ Sources: {len(result['sources'])}")

if result['sources']:
    print("\n📚 Sources used:")
    for s in result['sources']:
        print(f"  • {s['source']} ({s['score']:.0%})")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
