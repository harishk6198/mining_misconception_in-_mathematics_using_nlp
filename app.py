"""
Entry point for Hugging Face Spaces.
This file will be used by Hugging Face's default configuration.
It simply imports the Flask app and runs it.
"""

from huggingface_app import app

if __name__ == "__main__":
    # This file is used as the entry point for Hugging Face Spaces
    app.run(host="0.0.0.0", port=7860)