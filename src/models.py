from pydantic import BaseModel
from typing import List

class TextRequest(BaseModel):
    texts: List[str]  # Danh sách văn bản cần tạo embedding


# Model đại diện cho dữ liệu văn bản
class VanBan(BaseModel):
    id: int
    ten_van_ban: str
    ti_le_tuong_tu: float