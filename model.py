from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import logging
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MisconceptionModel:

    def __init__(self):
        # Use TF-IDF with better parameters for text classification
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            min_df=2,  # Require term to appear in at least 2 documents
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
            ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
            sublinear_tf=True,
            stop_words='english'
        )
        
        # Multi-label binarizer for handling multiple misconceptions per question
        self.mlb = MultiLabelBinarizer(sparse_output=False)
        
        # Use LogisticRegression which works better for text classification
        self.classifier = MultiOutputClassifier(
            LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                solver='liblinear',
                C=1.0
            )
        )
        
        self.is_fitted = False

    def prepare_features(self, data):
        """Create TF-IDF features from preprocessed text."""
        logger.info(f"Preparing features from {len(data)} samples")
        
        if data.empty:
            raise ValueError("Empty dataset provided")
        
        # Combine question text and correct answer for richer features
        combined_text = []
        for _, row in data.iterrows():
            text_parts = []
            
            # Add question text
            if 'processed_question' in row and pd.notna(row['processed_question']):
                text_parts.append(str(row['processed_question']))
            
            # Add correct answer
            if 'processed_correct_answer' in row and pd.notna(row['processed_correct_answer']):
                text_parts.append(str(row['processed_correct_answer']))
            
            # Add all answers if available
            if 'processed_all_answers' in row and pd.notna(row['processed_all_answers']):
                text_parts.append(str(row['processed_all_answers']))
            
            # Fallback to original text columns if processed ones don't exist
            if not text_parts:
                if 'QuestionText' in row and pd.notna(row['QuestionText']):
                    text_parts.append(str(row['QuestionText']))
                if 'CorrectAnswer' in row and pd.notna(row['CorrectAnswer']):
                    text_parts.append(str(row['CorrectAnswer']))
            
            combined_text.append(' '.join(text_parts) if text_parts else '')
        
        # Transform text to TF-IDF features
        if hasattr(self.vectorizer, 'vocabulary_'):
            # Model already fitted, transform only
            X = self.vectorizer.transform(combined_text)
        else:
            # First time, fit and transform
            X = self.vectorizer.fit_transform(combined_text)
        
        logger.info(f"Created feature matrix with shape: {X.shape}")
        return X

    def train(self, data):
        """Train model on initial data."""
        try:
            logger.info(f"Training model with {len(data)} samples")
            
            # Prepare features
            X = self.prepare_features(data)
            
            # Process misconception labels
            misconception_lists = []
            for i in range(len(data)):
                row = data.iloc[i]
                if 'misconceptions' in data.columns:
                    misconceptions = row['misconceptions']
                    if isinstance(misconceptions, list) and len(misconceptions) > 0:
                        # Filter out invalid values and convert to integers
                        valid_misconceptions = []
                        for m in misconceptions:
                            try:
                                if not pd.isna(m):
                                    valid_misconceptions.append(int(m))
                            except (ValueError, TypeError):
                                continue
                        misconception_lists.append(valid_misconceptions)
                    elif isinstance(misconceptions, (int, float)) and not pd.isna(misconceptions):
                        misconception_lists.append([int(misconceptions)])
                    else:
                        misconception_lists.append([])
                else:
                    misconception_lists.append([])
            
            # Get all unique misconception IDs
            all_misconceptions = set()
            for misconceptions in misconception_lists:
                all_misconceptions.update(misconceptions)
            
            if len(all_misconceptions) == 0:
                logger.error("No valid misconceptions found in training data")
                # Create a default "no misconception" class
                all_misconceptions = {0}
                misconception_lists = [[0] for _ in range(len(data))]
            
            logger.info(f"Found misconceptions: {sorted(list(all_misconceptions))}")
            
            # Create the multi-label binarizer with explicit classes
            sorted_misconceptions = sorted(list(all_misconceptions))
            self.mlb = MultiLabelBinarizer(classes=sorted_misconceptions, sparse_output=False)
            
            # Transform the misconception lists to binary matrix
            y = self.mlb.fit_transform(misconception_lists)
            
            logger.info(f"Label matrix shape: {y.shape}")
            logger.info(f"Classes: {self.mlb.classes_}")
            logger.info(f"Label distribution: {np.sum(y, axis=0).tolist()}")
            
            # Train the classifier
            self.classifier.fit(X, y)
            
            self.is_fitted = True
            logger.info("Model trained successfully")
            
            # Save model
            self._save_model()

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def predict(self, data):
        """Predict misconceptions for new data."""
        try:
            if not self.is_fitted:
                raise ValueError("Model not trained yet")

            logger.info(f"Generating predictions for {len(data)} samples")
            
            # Prepare features
            X = self.prepare_features(data)
            
            # Get prediction probabilities
            y_pred_proba = self.classifier.predict_proba(X)
            
            # Convert probabilities to predictions with threshold
            threshold = 0.3  # Lower threshold for better recall
            predictions = []
            
            for i in range(len(data)):
                sample_predictions = []
                
                # Check each misconception class
                for j, class_id in enumerate(self.mlb.classes_):
                    if j < len(y_pred_proba) and len(y_pred_proba[j]) > i:
                        # Get probability for positive class (misconception present)
                        if len(y_pred_proba[j][i]) > 1:
                            prob = y_pred_proba[j][i][1]  # Probability of class 1
                            if prob > threshold:
                                sample_predictions.append(int(class_id))
                
                # If no predictions above threshold, use the highest probability misconception
                if not sample_predictions:
                    max_prob = 0
                    best_misconception = 0
                    
                    for j, class_id in enumerate(self.mlb.classes_):
                        if j < len(y_pred_proba) and len(y_pred_proba[j]) > i:
                            if len(y_pred_proba[j][i]) > 1:
                                prob = y_pred_proba[j][i][1]
                                if prob > max_prob:
                                    max_prob = prob
                                    best_misconception = int(class_id)
                    
                    if max_prob > 0.1:  # Only if there's some confidence
                        sample_predictions.append(best_misconception)
                
                predictions.append(sample_predictions)
            
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def partial_fit(self, data):
        """Incrementally train the model with new data."""
        # For simplicity, retrain the entire model
        return self.train(data)

    def _save_model(self):
        """Save model artifacts to disk."""
        try:
            os.makedirs('models', exist_ok=True)
            
            # Save vectorizer
            joblib.dump(self.vectorizer, 'models/vectorizer.joblib')
            
            # Save multi-label binarizer
            joblib.dump(self.mlb, 'models/mlb.joblib')
            
            # Save classifier
            joblib.dump(self.classifier, 'models/classifier.joblib')
            
            # Save fitted status
            with open('models/model.joblib', 'w') as f:
                f.write('fitted')
            
            logger.info("Model artifacts saved successfully")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def _load_model(self):
        """Load model artifacts from disk."""
        try:
            if os.path.exists('models/model.joblib'):
                self.vectorizer = joblib.load('models/vectorizer.joblib')
                self.mlb = joblib.load('models/mlb.joblib')
                self.classifier = joblib.load('models/classifier.joblib')
                self.is_fitted = True
                logger.info("Model loaded successfully from disk")
                return True
            return False

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False


def evaluate_model(predictions, misconception_mapping):
    """Evaluate model predictions and return detailed results."""
    try:
        results = []
        
        # Load misconception mapping
        if isinstance(misconception_mapping, str):
            mapping_df = pd.read_csv(misconception_mapping)
        else:
            mapping_df = misconception_mapping
        
        misconception_dict = dict(zip(mapping_df['MisconceptionId'], mapping_df['MisconceptionName']))
        
        for i, pred_misconceptions in enumerate(predictions):
            result = {
                'question_number': i + 1,
                'predicted_misconceptions': pred_misconceptions,
                'misconception_details': []
            }
            
            # Get details for each predicted misconception
            for misconception_id in pred_misconceptions:
                if misconception_id in misconception_dict:
                    result['misconception_details'].append({
                        'id': misconception_id,
                        'name': misconception_dict[misconception_id]
                    })
                else:
                    result['misconception_details'].append({
                        'id': misconception_id,
                        'name': f'Unknown Misconception {misconception_id}'
                    })
            
            results.append(result)
        
        logger.info(f"Evaluation completed for {len(results)} predictions")
        return results

    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise