# ===== Modern, safe Dockerfile for credit-risk project =====

# 1️⃣ Use a stable Python base image
FROM python:3.11-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Avoid Python output buffering (helps logs in Docker)
ENV PYTHONUNBUFFERED=1

# 4️⃣ Install system dependencies safely
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpq-dev \
        curl \
        && rm -rf /var/lib/apt/lists/*

# 5️⃣ Copy requirements first (leverages Docker cache)
COPY requirements.txt .

# 6️⃣ Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7️⃣ Copy the rest of your project files
COPY . .

# 8️⃣ Expose the port your app will run on
EXPOSE 8000

# 9️⃣ Command to run your app (adjust if using Flask/FastAPI/Django)
CMD ["python", "-m", "src.api.main"]




# ===== End Dockerfile =====
