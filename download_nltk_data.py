import os
import logging
import nltk

def download_nltk_resources():
    """Download required NLTK resources."""
    logger = logging.getLogger(__name__)
    
    try:
        # Create nltk_data directory
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        
        # Download resources with explicit download_dir
        downloaded_resources = []
        
        for resource in ['stopwords', 'wordnet', 'punkt']:
            try:
                nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
                downloaded_resources.append(resource)
                logger.info(f"Successfully downloaded NLTK resource: {resource}")
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource {resource}: {str(e)}")
        
        # Check if all resources were downloaded
        return len(downloaded_resources) == 3
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {str(e)}")
        return False
