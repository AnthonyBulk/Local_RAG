import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)

required = [
    "streamlit",
    "langchain",
    "langchain_community",
    "langchain_text_splitters",
    "chromadb",
    "pydantic",
    "dotenv",
    "fastapi",
    "uvicorn",
    "sentence_transformers",
    "rank_bm25",
    "pypdf",   # pypdf
    "torch"
]

print("\nChecking required packages...")
missing = []
for pkg in required:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"❌ {pkg} not found")
        missing.append(pkg)

if not missing:
    print("\nAll required packages are installed!")
else:
    print("\nMissing packages:", missing)
    print("Tip: run `pip install -r requirements.txt` again inside your venv.")
