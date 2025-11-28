# Use official Python slim image
FROM python:3.11-slim

# Install deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

# Expose port used by Streamlit
ENV PORT=8080
ENV STREAMLIT_SERVER_RUN_ON_SAVE=false

EXPOSE ${PORT}

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "${PORT}", "--server.address", "0.0.0.0"]
