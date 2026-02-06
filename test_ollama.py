"""Quick test to check Ollama and phi3 installation"""

import ollama

print("Testing Ollama connection...\n")

try:
    # List all models
    result = ollama.list()
    print(f"Raw result: {result}\n")
    
    models = result.get('models', [])
    print(f"Number of models: {len(models)}\n")
    
    if models:
        print("Installed models:")
        for m in models:
            print(f"  - Name: {m.get('name', 'N/A')}")
            print(f"    Size: {m.get('size', 0) / 1e9:.2f} GB")
            print()
    else:
        print("No models found!")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
