# embedding.py
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch import Tensor

from database import get_db_connection

# Load model
tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-base')
model = AutoModel.from_pretrained('thenlper/gte-base')

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def generate_embeddings(text, metadata={}):
    combined_text = " ".join([text] + [v for k, v in metadata.items() if isinstance(v, str)])
    inputs = tokenizer(combined_text, return_tensors='pt', max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)

    attention_mask = inputs['attention_mask']
    embeddings = average_pool(outputs.last_hidden_state, attention_mask)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return json.dumps(embeddings.cpu().numpy().tolist()[0])


# Hàm lưu embeddings vào database
def save_embeddings(texts):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        for text in texts:
            embedding = generate_embeddings(text)
            cur.execute(
                "INSERT INTO van_ban (ten_van_ban, embedding) VALUES (%s, %s)", 
                (text, embedding)
            )
        conn.commit()
    finally:
        cur.close()
        conn.close()

# Hàm tìm kiếm văn bản gần nhất
def search_embeddings(query):
    query_embedding = generate_embeddings(query)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, ten_van_ban, 1 - (embedding <=> %s) AS cosine_similarity
            FROM van_ban
            ORDER BY cosine_similarity DESC LIMIT 5""",
        (query_embedding,)
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    return results