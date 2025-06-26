---
title: Mathematical Misconceptions Detector
emoji: 🧮
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: 3.9
app_port: 7860
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Mathematical Misconceptions Detector

## Overview
This application uses Natural Language Processing (NLP) and machine learning to detect and analyze mathematical misconceptions in student responses. It can identify common misconceptions, help educators understand learning gaps, and provide insights for targeted instruction.

## Features
- **Misconception Detection**: Analyze text responses to identify common mathematical misconceptions
- **Batch Analysis**: Process multiple responses at once for classroom-wide insights
- **Interactive Visualization**: View misconception patterns and frequencies
- **Model Training**: Ability to train on new data to improve detection accuracy

## How to Use
1. **Dashboard**: View statistics from previous analyses and model performance
2. **Analyze Responses**: Upload a CSV file of student responses for analysis
3. **View Results**: Review detected misconceptions with detailed explanations and correction strategies
4. **Scrape Content**: Analyze mathematical content from external websites
5. **Generate Predictions**: Test the model on the provided test dataset
6. **About**: Learn more about the project and mathematical misconceptions

## Technical Details
- Built with Flask, scikit-learn, and NLTK
- Uses TF-IDF vectorization for feature extraction
- Implements multi-label classification for misconception detection
- Processes mathematical notation in LaTeX format
- Includes web scraping capabilities for analyzing online content
- Features an interactive dashboard with data visualization
- Provides detailed misconception explanations with correction strategies

## Deployment

### Hugging Face Spaces
This application is designed to be easily deployed to Hugging Face Spaces. The package includes:

1. **Dockerfile** - For containerized deployment
2. **app.py** - Entry point for Hugging Face
3. **huggingface_requirements.txt** - Dependencies list

For detailed deployment instructions, please refer to the included `deploy_instructions.md` file.

### Local Deployment
To run the application locally:

1. Install the required dependencies: `pip install -r huggingface_requirements.txt`
2. Download NLTK data: `python download_nltk_data.py`
3. Run the Flask application: `python flask_app.py`
4. Access the app at `http://localhost:5000`