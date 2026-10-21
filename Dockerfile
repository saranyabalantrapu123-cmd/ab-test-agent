FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python scripts/generate_experiments.py

EXPOSE 8501
# Offline demo mode unless ANTHROPIC_API_KEY (or another provider key) is passed in.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true"]
