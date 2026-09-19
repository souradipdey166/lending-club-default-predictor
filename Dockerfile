FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (better Docker layer caching - only
# reinstalls when requirements.txt actually changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

EXPOSE 8000

# Run the API, using the same --app-dir approach that worked locally
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]