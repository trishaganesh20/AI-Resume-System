import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to your .env file.")
    return OpenAI(api_key=key)

def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    """
    Returns embeddings for a list of texts.
    """
    client = get_client()
    resp = client.embeddings.create(
        model=model,
        input=texts
    )
    return [d.embedding for d in resp.data]
