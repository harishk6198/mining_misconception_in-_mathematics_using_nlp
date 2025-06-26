FROM python:3.11-slim

WORKDIR /app

# Copy the requirements file
COPY huggingface_requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r huggingface_requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# Copy all application files
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p models results uploads static/css static/js templates

# Expose the port
EXPOSE 7860

# Run the app
CMD ["python", "flask_app.py"]