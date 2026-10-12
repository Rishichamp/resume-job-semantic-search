"""Entry point. Usage: python run.py"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"\n🔍  Resume-to-Job Semantic Search Engine")
    print(f"   Embedding  : {settings.EMBEDDING_PROVIDER} / {settings.EMBEDDING_MODEL}")
    print(f"   Chroma DB  : {settings.CHROMA_PATH}")
    print(f"   PII redact : {settings.REDACT_PII}")
    print(f"   Docs       : http://127.0.0.1:{settings.APP_PORT}/docs")
    print(f"\n   After starting, load sample data:")
    print(f"   python sample_data/load_samples.py\n")
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT,
                reload=settings.DEBUG, log_level="info")
