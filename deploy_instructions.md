# Math Misconception Detector - Deployment Instructions

This document provides instructions for deploying the Math Misconception Detector application to Hugging Face Spaces.

## About the Application

The Math Misconception Detector is an advanced educational platform that leverages NLP and machine learning to identify and address mathematical misconceptions in student responses. The application analyzes text responses to detect specific misconceptions, providing detailed explanations and suggested correction strategies.

## Deployment to Hugging Face Spaces

### Option 1: Using the Hugging Face UI

1. Create a new Space on Hugging Face:
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Select "Docker" as the Space SDK
   - Fill in the Space name, e.g., "math-misconception-detector"
   - Set the visibility (Public or Private)
   - Click "Create Space"

2. Upload files to the Space:
   - Use the Hugging Face UI to upload all the project files
   - Make sure to include all required files: Python scripts, CSV data files, templates, static files, etc.
   - Upload the `Dockerfile` and `huggingface_requirements.txt`

3. The Space will automatically build and deploy the application.

### Option 2: Using Git and the Hugging Face CLI

1. Install the Hugging Face CLI:
   ```bash
   pip install huggingface_hub
   ```

2. Login to Hugging Face:
   ```bash
   huggingface-cli login
   ```

3. Create a new repository for your Space:
   ```bash
   huggingface-cli repo create math-misconception-detector --type space --sdk docker
   ```

4. Clone the repository:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/math-misconception-detector
   ```

5. Copy all project files to the cloned repository folder.

6. Push to Hugging Face:
   ```bash
   cd math-misconception-detector
   git add .
   git commit -m "Initial commit"
   git push
   ```

## Important Files for Deployment

- `huggingface_app.py`: Entry point for the application on Hugging Face
- `huggingface_requirements.txt`: Dependencies for the application
- `Dockerfile`: Instructions for building the Docker container
- `flask_app.py`: Main Flask application
- `model.py`, `preprocessing.py`, `utils.py`: Core application logic
- `templates/` and `static/`: UI components

## After Deployment

After deployment, the application will automatically train the model (if no pre-trained model is found) and be ready to use. Users can:

1. Navigate to the Dashboard to see statistics from previous analyses
2. Upload CSV files with student responses on the Analyze page 
3. View detailed results with misconception explanations
4. Use the web scraper to analyze math content from external websites

## Troubleshooting

- If you encounter issues with NLTK data, ensure the required NLTK packages are downloaded in the Dockerfile
- Check that all necessary directories exist: `models/`, `results/`, `uploads/`, etc.
- Verify that the port configuration in `app.py` and `huggingface_app.py` matches Hugging Face's expectations (port 7860)
- If you encounter "Address already in use" errors, confirm that only one app is trying to use port 7860
- For memory issues, consider reducing the batch size in the model training process

## Model Training

The application will automatically train the model on first run if no pre-trained model is found. To pre-train the model:

1. Ensure `train.csv` is included in the deployment
2. The application will train the model on startup
3. Trained model files will be saved to the `models/` directory