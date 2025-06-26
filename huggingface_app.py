"""
Modified entry point for Hugging Face Spaces deployment.
This file imports the Flask app and sets it up to run on the Hugging Face Spaces platform.
"""

import os
import nltk
from flask_app import app

# Download required NLTK data if not already present
def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        # Check if punkt exists, if not download all necessary resources
        if not os.path.exists(os.path.join(nltk.data.path[0], 'tokenizers', 'punkt')):
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            print("NLTK resources downloaded successfully")
        else:
            print("NLTK resources already exist")
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")

# Download NLTK resources on startup
download_nltk_resources()

# Configure app for Hugging Face Spaces
if __name__ == '__main__':
    # Hugging Face Spaces uses port 7860
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)