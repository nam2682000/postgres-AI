# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from routes import router



app = FastAPI()
# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả domain
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả HTTP methods (GET, POST, PUT, DELETE, ...)
    allow_headers=["*"],  # Cho phép tất cả headers
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
