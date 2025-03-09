# embedding.py
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import re
import unicodedata
from database import get_db_connection
from models import VanBan

# Load multilingual-e5-large
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")


def average_pool(last_hidden_states, attention_mask):
    """ Hàm pooling để lấy vector trung bình từ token embeddings """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def normalize_text(text):
    """Chuẩn hóa văn bản trước khi tạo embeddings"""
    text = text.lower().strip()  # Chuyển về chữ thường, xóa khoảng trắng thừa
    text = unicodedata.normalize("NFKC", text)  # Chuẩn hóa Unicode

    # Loại bỏ ký tự không cần thiết, giữ lại chữ, số và dấu câu quan trọng
    text = re.sub(r"[^a-zA-Z0-9àáảãạăắằẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ.,!?()]", " ", text)
    
    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip()

def generate_embeddings(text, mode="passage"):
    """ Sinh embeddings cho văn bản đầu vào """
    # Định dạng câu theo chuẩn của E5 (tiếng Việt vẫn dùng format này)
    text = normalize_text(text)
    formatted_text = f"{mode}: {text}"  # Nếu là văn bản index thì dùng "passage: {text}"

    # Tokenize văn bản
    inputs = tokenizer(formatted_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)

    # Áp dụng pooling để lấy vector nhúng
    embeddings = average_pool(outputs.last_hidden_state, inputs["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)  # Chuẩn hóa vector

    return json.dumps(embeddings.cpu().numpy().tolist()[0])  # Trả về dưới dạng JSON


# Hàm lưu embeddings vào database
def save_embeddings(texts):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        for text in texts:
            embedding = generate_embeddings(text)
            cur.execute(
                "INSERT INTO van_ban (ten_van_ban, embedding) VALUES (%s, %s)",
                (text, embedding),
            )
        conn.commit()
    finally:
        cur.close()
        conn.close()


# Hàm tìm kiếm văn bản gần nhất
def search_embeddings(text):
    conn = get_db_connection()
    cur = conn.cursor()
    if text:
        query_embedding = generate_embeddings(text,"query")
        sql_text = """SELECT id, ten_van_ban, 1 - (embedding <=> %s) AS ti_le_tuong_tu
                      FROM van_ban
                      ORDER BY ti_le_tuong_tu DESC LIMIT 5"""
        cur.execute(sql_text, (query_embedding,),)
    else:
        sql_text = "SELECT id, ten_van_ban, 0.0 AS ti_le_tuong_tu FROM van_ban"
        cur.execute(sql_text)

    results = cur.fetchall()
    cur.close()
    conn.close()
    
    return [VanBan(id=row[0], ten_van_ban=row[1], ti_le_tuong_tu = row[2]) for row in results]
