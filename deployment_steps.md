# Deployment Steps for Hugging Face Spaces

## Quick Deployment Steps

1. **Download the Deployment Package**
   - Locate the `math_misconception_detector_hf.tar.gz` file in your project directory
   - Download this file to your local computer

2. **Create a New Hugging Face Space**
   - Go to [https://huggingface.co/spaces](https://huggingface.co/spaces)
   - Sign in to your Hugging Face account (or create one if needed)
   - Click "Create new Space"
   - Choose a name for your space (e.g., "math-misconception-detector")
   - Select "Docker" as the SDK
   - Set the hardware to "CPU" (free tier)
   - Click "Create Space"

3. **Upload Files**
   - In your new Space, click the "Files" tab
   - Click "Upload files"
   - Extract your `math_misconception_detector_hf.tar.gz` file locally
   - Upload all the extracted files to your Space
   - Alternatively, use Git to push the files to your Space repository

4. **Wait for Deployment**
   - Hugging Face will automatically build and deploy your application
   - This process may take a few minutes
   - You can check the build logs by clicking on the "Factory" tab

5. **Access Your Application**
   - Once the build is complete, click on the "App" tab
   - Your application will be running at `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

## Important Files

- **app.py**: Entry point for Hugging Face
- **Dockerfile**: Configuration for the Docker container
- **huggingface_requirements.txt**: Dependencies for the application
- **README.md**: Contains the Space configuration and application description

## Troubleshooting

If you encounter any issues during deployment:

1. Check the "Factory" tab for build logs
2. Ensure all files were properly uploaded
3. Verify that your Space has the correct configuration in README.md
4. Confirm that the port in app.py matches the port in the Hugging Face configuration (7860)

For more detailed instructions, refer to the included `deploy_instructions.md` file.