FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Ensure the port is 7860
EXPOSE 7860

# Start uvicorn directly
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
