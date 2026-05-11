FROM python:3.9-slim

WORKDIR /app

# Install dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=src/app.py
ENV FLASK_ENV=development
ENV FLASK_RUN_HOST=0.0.0.0

# Run the application
CMD ["flask", "run"]
