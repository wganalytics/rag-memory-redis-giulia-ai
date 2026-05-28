from src.core.rag_engine import rag_engine_instance
import json

count = rag_engine_instance.vectorstore._collection.count()
print(f"Total documentos na base (chunks): {count}")

if count > 0:
    results = rag_engine_instance.vectorstore._collection.get(limit=100)
    sources = set()
    for meta in results['metadatas']:
        sources.add(meta.get('source'))
    
    print("\nFontes encontradas na base:")
    for s in sources:
        print(f"- {s}")
