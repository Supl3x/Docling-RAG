"""GPU Status Diagnostic - Check if everything can use GPU"""

print("="*60)
print("GPU DIAGNOSTIC CHECK")
print("="*60)

# 1. Check PyTorch CUDA
print("\n1. PyTorch CUDA Status:")
try:
    import torch
    print(f"   PyTorch installed: ✓ v{torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA available: {'✓ YES' if cuda_available else '✗ NO (CPU only)'}")
    
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
    else:
        print("   ⚠️  PROBLEM: PyTorch is CPU-only!")
        print("   FIX: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
except ImportError:
    print("   ✗ PyTorch not installed")

# 2. Check SentenceTransformers
print("\n2. SentenceTransformers Status:")
try:
    from sentence_transformers import SentenceTransformer
    print("   SentenceTransformers installed: ✓")
    
    if 'torch' in dir():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Device will use: {device}")
        
        if device == 'cpu':
            print("   ⚠️  WARNING: Embeddings will run on CPU (SLOW)")
except ImportError:
    print("   ✗ SentenceTransformers not installed")

# 3. Check Ollama
print("\n3. Ollama Status:")
try:
    import ollama
    result = ollama.list()
    models = result.models if hasattr(result, 'models') else []
    
    print(f"   Ollama running: ✓")
    print(f"   Models installed: {len(models)}")
    
    for m in models:
        model_name = getattr(m, 'model', 'Unknown')
        print(f"     - {model_name}")
    
    print("\n   Ollama GPU Status:")
    print("   Ollama automatically uses GPU if NVIDIA driver + CUDA are installed")
    print("   Check Task Manager > Performance > GPU while chatting to verify")
    
except Exception as e:
    print(f"   ✗ Ollama error: {e}")

# 4. Check NVIDIA
print("\n4. NVIDIA Driver Status:")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   NVIDIA driver: ✓ Installed")
        # Extract GPU name
        for line in result.stdout.split('\n'):
            if 'NVIDIA' in line or 'GeForce' in line or 'RTX' in line:
                print(f"   {line.strip()}")
                break
    else:
        print("   ✗ nvidia-smi not found")
except:
    print("   ⚠️  Could not check NVIDIA driver")

# Summary
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)

if 'torch' in dir() and torch.cuda.is_available():
    print("✓ System is GPU-READY for embeddings")
else:
    print("✗ EMBEDDINGS WILL RUN ON CPU (Install PyTorch CUDA)")

print("\nFor Ollama to use GPU:")
print("  1. NVIDIA driver must be installed (check above)")
print("  2. PyTorch CUDA not required for Ollama")
print("  3. Ollama auto-detects GPU")
print("  4. Watch Task Manager > GPU while chatting to verify")

print("\n" + "="*60)
