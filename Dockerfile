# Sử dụng Python 3.12 làm base image
FROM python:3.12

# Đặt thư mục làm thư mục làm việc trong container
WORKDIR /app

# Copy tất cả file vào thư mục làm việc
COPY . /app

# Cài đặt các dependency
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Mở cổng 8000
EXPOSE 8000

# Chạy ứng dụng FastAPI bằng Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
