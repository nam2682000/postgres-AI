from typing import List, Optional
from fastapi import APIRouter
from models import TextRequest, VanBan
from services import save_embeddings, search_embeddings

router = APIRouter()


@router.post("/embed")
def create_embedding(request: TextRequest):
    save_embeddings(request.texts)
    return {"message": "Embeddings saved!"}


@router.get("/search", response_model=List[VanBan])
def search_embedding(query: Optional[str] = None):  
    results = search_embeddings(query)
    return results
