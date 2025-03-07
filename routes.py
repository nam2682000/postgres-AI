from fastapi import APIRouter
from models import TextRequest
from services import save_embeddings, search_embeddings

router = APIRouter()

@router.post("/embed")
def create_embedding(request: TextRequest):
    save_embeddings(request.texts)
    return {"message": "Embeddings saved!"}

@router.get("/search")
def search_embedding(query: str):
    results = search_embeddings(query)
    return {"query": query, "results": results}
