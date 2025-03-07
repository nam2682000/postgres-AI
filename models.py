from pydantic import BaseModel
from typing import List

class TextRequest(BaseModel):
    texts: List[str]  # Danh sách văn bản cần tạo embedding
