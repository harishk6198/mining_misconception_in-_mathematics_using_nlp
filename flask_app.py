import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
from io import BytesIO, StringIO

# Import our model and utility functions
from model import MisconceptionModel
from preprocessing import preprocess_data
from utils import save_results
from webscraper import create_scraper_route

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store session data
model = MisconceptionModel()
# Check if model files exist and load them
model_dir = 'models'
model_file = os.path.join(model_dir, 'model.joblib')
if os.path.exists(model_file):
    print(f"Found model file: {model_file}")
    try:
        # Try to load the model
        print("Loading model from file...")
        model._load_model()
        model.is_fitted = True
        print("Model loaded successfully and marked as fitted")
    except Exception as e:
        print(f"Error loading model: {e}")
        model.is_fitted = False
else:
    print("No model file found, model not fitted")
    model.is_fitted = False

misconception_mapping = {}
misconception_explanations = {}
analysis_results = None
model_trained = model.is_fitted  # Track if model has been trained

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_misconception_mapping():
    global misconception_mapping
    # Load misconception mapping from CSV
    try:
        mapping_df = pd.read_csv('misconception_mapping.csv')
        # Check if the column exists, if not try alternative name
        if 'Description' in mapping_df.columns:
            misconception_mapping = dict(zip(mapping_df['MisconceptionId'], mapping_df['Description']))
        elif 'MisconceptionName' in mapping_df.columns:
            misconception_mapping = dict(zip(mapping_df['MisconceptionId'], mapping_df['MisconceptionName']))
        else:
            print("Error: Could not find Description or MisconceptionName column in misconception_mapping.csv")
            return False
        return True
    except Exception as e:
        print(f"Error loading misconception mapping: {e}")
        return False

def load_misconception_explanations():
    global misconception_explanations
    # Load misconception explanations from CSV
    try:
        explanations_df = pd.read_csv('misconception_explanations.csv')
        # Create a dictionary with misconception ID as key and explanation details as value
        for _, row in explanations_df.iterrows():
            misconception_id = row['MisconceptionId']
            misconception_explanations[misconception_id] = {
                'explanation': row['Explanation'],
                'example': row['Example'],
                'correction_strategy': row['CorrectionStrategy']
            }
        print(f"Loaded {len(misconception_explanations)} misconception explanations")
        return True
    except Exception as e:
        print(f"Error loading misconception explanations: {e}")
        return False

def get_dashboard_stats():
    """Get stats for dashboard display."""
    # Default values
    stats = {
        'analyzed': 0,
        'misconceptions': 0,
        'with_misconceptions': 0,
        'no_misconceptions': 0
    }
    
    # If we have analysis results, compute stats
    global analysis_results
    
    # Try to load results from file if not available in memory
    if not analysis_results:
        results_file = os.path.join('results', 'misconceptions.json')
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    analysis_results = json.load(f)
                    print(f"Loaded {len(analysis_results)} results from file for dashboard")
            except Exception as e:
                print(f"Error loading results from file for dashboard: {e}")
    
    if analysis_results:
        stats['analyzed'] = len(analysis_results)
        stats['with_misconceptions'] = sum(1 for r in analysis_results if len(r['misconception_ids']) > 0)
        stats['no_misconceptions'] = stats['analyzed'] - stats['with_misconceptions']
        stats['misconceptions'] = sum(len(r['misconception_ids']) for r in analysis_results)
    
    return stats

def train_model():
    """Train model with training data."""
    global model, model_trained
    
    try:
        # Load and preprocess training data
        train_df = pd.read_csv('train.csv')
        print(f"Loaded training data with shape: {train_df.shape}")
        
        # Process data in chunks to handle large datasets
        chunk_size = 500
        num_chunks = (len(train_df) + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(train_df))
            chunk = train_df.iloc[start_idx:end_idx].copy()
            
            print(f"Processing chunk {i+1}/{num_chunks} (rows {start_idx}-{end_idx})")
            
            # Preprocess the chunk
            processed_data = preprocess_data(chunk, is_training=True)
            
            print(f"Preprocessed chunk {i+1}, shape: {processed_data.shape}")
            
            # Display sample misconceptions for a few rows
            for j in range(min(5, len(processed_data))):
                print(f"Sample {j} misconceptions: {processed_data.iloc[j]['misconceptions']}")
            
            # Train the model with the chunk
            if i == 0:
                print("Starting initial model training...")
                model.train(processed_data)
            else:
                print(f"Performing partial fit for chunk {i+1}")
                model.partial_fit(processed_data)
            
            print(f"Completed partial fit for chunk {i+1}")
        
        model_trained = True
        return True
    except Exception as e:
        print(f"Error training model: {e}")
        return False

def generate_test_predictions():
    """Generate predictions for test dataset."""
    global model, misconception_mapping
    
    if not hasattr(model, 'is_fitted') or not model.is_fitted:
        print("Model is not trained yet. Please train the model first.")
        flash("Model is not trained yet. Please train the model first.", "error")
        return None
    
    try:
        # Load test data
        test_df = pd.read_csv('test.csv')
        print(f"Loaded test data with shape: {test_df.shape}")
        
        # Preprocess test data
        processed_data = preprocess_data(test_df, is_training=False)
        print(f"Preprocessed test data with shape: {processed_data.shape}")
        
        # Generate predictions
        predictions = model.predict(processed_data)
        print(f"Generated predictions for {len(predictions)} test instances")
        
        # Format predictions for submission
        submission = pd.DataFrame({
            'QuestionId': test_df['QuestionId'].values,
            'MisconceptionId': [' '.join(map(str, p)) for p in predictions]
        })
        
        return submission
    except Exception as e:
        import traceback
        print(f"Error generating test predictions: {e}")
        print(traceback.format_exc())
        flash(f"Error generating test predictions: {str(e)}", "error")
        return None

def analyze_file(file):
    """Analyze uploaded file to detect misconceptions."""
    global model, misconception_mapping, misconception_explanations, analysis_results
    
    print("=== Starting file analysis ===")
    
    if not model.is_fitted:
        print("Model not fitted, cannot analyze file")
        flash("Model is not trained yet. Please train the model first.", "error")
        return False
    
    try:
        print(f"Analyzing file: {file}")
        # Read uploaded file
        try:
            print(f"Reading CSV file: {file}")
            df = pd.read_csv(file)
            print(f"Successfully read file with shape: {df.shape}")
            # Print column names for debugging
            print(f"Columns in uploaded file: {df.columns.tolist()}")
            if len(df) > 0:
                # Sample first row for debugging
                print(f"First row sample: {df.iloc[0].to_dict()}")
        except Exception as read_error:
            print(f"Error reading file: {read_error}")
            flash(f"Error reading file: {str(read_error)}", "error")
            return False
        print(f"Loaded analysis data with shape: {df.shape}")
        
        # Basic validation of file format
        required_columns = ['question_id', 'question_text']
        if not all(col in df.columns for col in required_columns):
            # Try alternative column names (QuestionId, QuestionText)
            alt_required_columns = ['QuestionId', 'QuestionText']
            if not all(col in df.columns for col in alt_required_columns):
                flash(f"Invalid file format. File must contain columns: question_id, question_text", "error")
                return False
            else:
                # Rename columns to standardized format
                rename_dict = {}
                if 'QuestionId' in df.columns:
                    rename_dict['QuestionId'] = 'question_id'
                if 'QuestionText' in df.columns:
                    rename_dict['QuestionText'] = 'question_text'
                if 'CorrectAnswer' in df.columns:
                    rename_dict['CorrectAnswer'] = 'correct_answer'
                
                df = df.rename(columns=rename_dict)
        
        # Ensure correct_answer column exists
        if 'correct_answer' not in df.columns:
            df['correct_answer'] = 'Not provided'
            
        # Preprocess data for analysis
        try:
            processed_data = preprocess_data(df, is_training=False)
            print(f"Preprocessed analysis data with shape: {processed_data.shape}")
        except Exception as preprocess_error:
            print(f"Error in preprocessing: {preprocess_error}")
            flash(f"Error preprocessing data: {str(preprocess_error)}", "error")
            return False
        
        # Generate predictions
        try:
            predictions = model.predict(processed_data)
            print(f"Generated predictions for {len(predictions)} instances")
        except Exception as predict_error:
            print(f"Error in prediction: {predict_error}")
            flash(f"Error generating predictions: {str(predict_error)}", "error")
            return False
        
        # Format results
        results = []
        for i in range(min(len(predictions), len(df))):
            try:
                # Get the question details
                q_id = df.iloc[i]['question_id']
                q_text = df.iloc[i]['question_text']
                
                # Get correct answer
                correct_answer = df.iloc[i]['correct_answer']
                
                # Get misconception information
                misconception_details = []
                pred = predictions[i] if i < len(predictions) else []
                
                for mid in pred:
                    # Convert numpy types to Python types if needed
                    if hasattr(mid, 'item'):
                        mid = mid.item()
                    
                    # Get the misconception name
                    name = misconception_mapping.get(mid, f"Unknown Misconception ({mid})")
                    
                    # Get explanation, example, and correction strategy if available
                    explanation = None
                    example = None
                    correction_strategy = None
                    
                    if mid in misconception_explanations:
                        explanation = misconception_explanations[mid]['explanation']
                        example = misconception_explanations[mid]['example']
                        correction_strategy = misconception_explanations[mid]['correction_strategy']
                    
                    # Add to details
                    misconception_details.append({
                        'id': mid,
                        'name': name,
                        'explanation': explanation,
                        'example': example,
                        'correction_strategy': correction_strategy
                    })
                
                # Create result entry with all information
                results.append({
                    'question_id': q_id,
                    'question_text': q_text,
                    'correct_answer': correct_answer,
                    'misconception_ids': pred,
                    'misconception_details': misconception_details
                })
            except Exception as format_error:
                print(f"Error formatting result for item {i}: {format_error}")
                # Continue with the next item instead of failing completely
                continue
        
        if not results:
            flash("No valid results were generated. Please check your file format.", "error")
            return False
            
        analysis_results = results
        save_success = save_results(results)
        if not save_success:
            flash("Results generated but could not be saved to file", "warning")
            # Continue anyway since we have the results in memory
        
        return True
    except Exception as e:
        print(f"Error analyzing file: {e}")
        flash(f"Error analyzing file: {str(e)}", "error")
        return False

# Custom filter to convert list to JSON
@app.template_filter('to_json')
def to_json(value):
    return json.dumps(value)

@app.route('/')
def index():
    """Home page with dashboard."""
    global model, misconception_mapping, model_trained
    
    # Load misconception mapping if not loaded
    if not misconception_mapping:
        load_misconception_mapping()
    
    # Don't auto-train model on startup to improve responsiveness
    # We'll let the user click the train button instead
    model_status = "Not Trained ❌" 
    if not model_trained and hasattr(model, 'is_fitted') and model.is_fitted:
        model_status = "Trained and Ready ✅"
    
    # Get dashboard stats
    stats = get_dashboard_stats()
    
    # Get sample misconception categories
    sample_misconceptions = []
    if misconception_mapping:
        sample_size = min(10, len(misconception_mapping))
        sample_ids = list(misconception_mapping.keys())[:sample_size]
        
        for mid in sample_ids:
            sample_misconceptions.append({
                'id': mid,
                'description': misconception_mapping[mid]
            })
    
    # Check if model is trained
    model_status = "Trained and Ready ✅" if hasattr(model, 'is_fitted') and model.is_fitted else "Not Trained ❌"
    
    return render_template('index.html', 
                           stats=stats, 
                           model_status=model_status,
                           sample_misconceptions=sample_misconceptions,
                           results=analysis_results,
                           active_tab='dashboard')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Analyze page for file upload and analysis."""
    global model, misconception_mapping, analysis_results
    
    # Load misconception mapping if not loaded
    if not misconception_mapping:
        load_misconception_mapping()
    
    if request.method == 'POST':
        try:
            print("=== POST request to /analyze ===")
            # Check if the post request has the file part
            if 'file' not in request.files:
                print("No file part in request")
                flash('No file part in request', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            print(f"File received: {file.filename}")
            
            # If user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                print("No selected file (empty filename)")
                flash('No selected file', 'error')
                return redirect(request.url)
            
            if file and allowed_file(file.filename):
                print(f"File allowed: {file.filename}")
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Ensure upload directory exists
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                print(f"Saving file to: {file_path}")
                file.save(file_path)
                
                # Analyze the file
                print(f"Starting file analysis for: {file_path}")
                success = analyze_file(file_path)
                
                if success:
                    print("Analysis completed successfully!")
                    flash('File analyzed successfully!', 'success')
                    return redirect(url_for('results'))
                else:
                    print("Analysis failed")
                    flash('Analysis failed. Please check logs for details.', 'error')
                    return redirect(request.url)
            else:
                print(f"File not allowed: {file.filename}")
                flash(f'File type not allowed. Please upload a CSV file.', 'error')
                return redirect(request.url)
        except Exception as e:
            print(f"Unexpected error in analyze route: {e}")
            flash(f'An unexpected error occurred: {str(e)}', 'error')
            return redirect(request.url)
    
    # Check if model is trained
    model_status = "Trained and Ready ✅" if hasattr(model, 'is_fitted') and model.is_fitted else "Not Trained ❌"
    
    return render_template('analyze.html', 
                           model_status=model_status,
                           active_tab='analyze')

@app.route('/results')
def results():
    """Results page to display analysis results."""
    global analysis_results, misconception_mapping, misconception_explanations
    
    try:
        # If analysis_results is not available in memory, try to load from file
        if not analysis_results:
            print("No results in memory, trying to load from file...")
            results_file = os.path.join('results', 'misconceptions.json')
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        analysis_results = json.load(f)
                        print(f"Successfully loaded {len(analysis_results)} results from file")
                except Exception as e:
                    error_msg = f"Error loading results from file: {e}"
                    print(error_msg)
                    flash(error_msg, 'error')
                    return render_template('results.html', results=None, active_tab='results')
            else:
                print("Results file not found")
        
        if not analysis_results:
            flash('No analysis results available. Please analyze a file first.', 'warning')
            return render_template('results.html', results=None, active_tab='results')
    except Exception as e:
        error_msg = f"Unexpected error in results route initialization: {e}"
        print(error_msg)
        flash(error_msg, 'error')
        return render_template('results.html', results=None, active_tab='results')
    
    # Get summary stats
    total_analyzed = len(analysis_results)
    with_misconceptions = sum(1 for r in analysis_results if len(r['misconception_ids']) > 0)
    total_misconceptions = sum(len(r['misconception_ids']) for r in analysis_results)
    
    # Count occurrences of each misconception
    misconception_counts = {}
    for result in analysis_results:
        for mid in result['misconception_ids']:
            if hasattr(mid, 'item'):  # Convert numpy types to Python types
                mid = mid.item()
            if mid not in misconception_counts:
                misconception_counts[mid] = 0
            misconception_counts[mid] += 1
    
    # Sort misconceptions by frequency
    sorted_misconceptions = sorted(misconception_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Get top misconceptions with descriptions
    top_misconceptions = []
    for mid, count in sorted_misconceptions[:5]:
        description = misconception_mapping.get(mid, f"Unknown Misconception ({mid})")
        explanation = None
        example = None
        correction_strategy = None
        
        if mid in misconception_explanations:
            explanation = misconception_explanations[mid]['explanation']
            example = misconception_explanations[mid]['example']
            correction_strategy = misconception_explanations[mid]['correction_strategy']
        
        top_misconceptions.append({
            'id': mid,
            'description': description,
            'count': count,
            'explanation': explanation,
            'example': example,
            'correction_strategy': correction_strategy
        })
    
    # Create chart data for misconception distribution
    misconception_labels = []
    misconception_counts_list = []
    for mid, count in sorted_misconceptions[:10]:  # Top 10 for chart
        description = misconception_mapping.get(mid, f"Unknown ({mid})")
        if len(description) > 20:
            description = description[:20] + "..."
        misconception_labels.append(f"{mid}: {description}")
        misconception_counts_list.append(count)
    
    # Prepare detailed analysis data
    details = []
    for result in analysis_results:
        try:
            # Get question details
            q_id = result['question_id']
            
            # Get question text and answer if available
            q_text = result.get('question_text', 'Question text not available')
            c_answer = result.get('correct_answer', 'Answer not available')
            
            # Get misconception details from result or generate them
            misconception_info = []
            
            # If we have misconception_details in the result, use those
            if 'misconception_details' in result and result['misconception_details']:
                misconception_info = result['misconception_details']
            else:
                # Otherwise, build them from misconception_ids
                for mid in result.get('misconception_ids', []):
                    if hasattr(mid, 'item'):  # Convert numpy types to Python types
                        mid = mid.item()
                    
                    # Get the name and additional info if available
                    name = misconception_mapping.get(mid, f"Unknown Misconception ({mid})")
                    explanation = None
                    example = None
                    correction_strategy = None
                    
                    if mid in misconception_explanations:
                        explanation = misconception_explanations[mid]['explanation']
                        example = misconception_explanations[mid]['example']
                        correction_strategy = misconception_explanations[mid]['correction_strategy']
                    
                    misconception_info.append({
                        'id': mid,
                        'name': name,
                        'explanation': explanation,
                        'example': example,
                        'correction_strategy': correction_strategy
                    })
        except Exception as e:
            print(f"Error processing result {result}: {e}")
            continue
        
        details.append({
            'question_id': q_id,
            'question_text': q_text,
            'correct_answer': c_answer,
            'misconception_details': misconception_info
        })
    
    # Prepare complete results object
    formatted_results = {
        'total_analyzed': total_analyzed,
        'with_misconceptions': with_misconceptions,
        'total_misconceptions': total_misconceptions,
        'misconception_counts': misconception_counts_list,
        'misconception_labels': misconception_labels,
        'top_misconceptions': top_misconceptions,
        'details': details
    }
    
    return render_template('results.html', 
                           results=formatted_results,
                           active_tab='results')

@app.route('/about')
def about():
    """About page with information about the application."""
    return render_template('about.html', active_tab='about')

@app.route('/train', methods=['POST'])
def train():
    """API endpoint to train the model."""
    success = train_model()
    
    if success:
        return jsonify({'status': 'success', 'message': 'Model trained successfully!'})
    else:
        return jsonify({'status': 'error', 'message': 'Error training model. Check logs for details.'})

@app.route('/generate-test-predictions', methods=['POST'])
def predict_test():
    """API endpoint to generate predictions for test dataset."""
    try:
        # Check if model is trained
        if not hasattr(model, 'is_fitted') or not model.is_fitted:
            return jsonify({
                'status': 'error', 
                'message': 'Model is not trained yet. Please train the model first.'
            }), 400
            
        submission_df = generate_test_predictions()
        
        if submission_df is not None:
            # Create a CSV string from the dataframe
            csv_buffer = StringIO()
            submission_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            # Create a BytesIO object for the file download
            mem = BytesIO()
            mem.write(csv_buffer.getvalue().encode('utf-8'))
            mem.seek(0)
            
            return send_file(
                mem,
                as_attachment=True,
                download_name='submission.csv',
                mimetype='text/csv'
            )
        else:
            # Error already flashed in generate_test_predictions
            return jsonify({
                'status': 'error', 
                'message': 'Error generating predictions. Please check the logs for details.'
            }), 500
    except Exception as e:
        import traceback
        print(f"Unexpected error in predict_test route: {e}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error', 
            'message': f'Unexpected error: {str(e)}'
        }), 500

@app.route('/download-results', methods=['GET'])
def download_results():
    """API endpoint to download analysis results."""
    global analysis_results
    
    # If analysis_results is not available in memory, try to load from file
    if not analysis_results:
        results_file = os.path.join('results', 'misconceptions.json')
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    analysis_results = json.load(f)
                    print(f"Loaded {len(analysis_results)} results from file for download")
            except Exception as e:
                print(f"Error loading results from file: {e}")
    
    if not analysis_results:
        return jsonify({'status': 'error', 'message': 'No analysis results available.'})
    
    # Convert results to JSON-serializable format
    json_serializable_results = []
    for result in analysis_results:
        # Create a new dict with serializable values
        clean_result = {
            'question_id': int(result['question_id']) if isinstance(result['question_id'], (np.int64, np.int32)) else result['question_id'],
            'question_text': result.get('question_text', 'Question text not available'),
            'correct_answer': result.get('correct_answer', 'Answer not available'),
            'misconception_ids': [],
            'misconception_details': result.get('misconception_details', [])
        }
        
        # Convert misconception_ids from numpy types to regular Python ints
        for mid in result['misconception_ids']:
            if hasattr(mid, 'item'):  # Check if it's a numpy type
                clean_result['misconception_ids'].append(mid.item())
            else:
                clean_result['misconception_ids'].append(mid)
        
        json_serializable_results.append(clean_result)
    
    # Create JSON data
    results_json = json.dumps(json_serializable_results, indent=2)
    
    # Create a BytesIO object for the file download
    mem = BytesIO()
    mem.write(results_json.encode('utf-8'))
    mem.seek(0)
    
    return send_file(
        mem,
        as_attachment=True,
        download_name='misconception_results.json',
        mimetype='application/json'
    )

@app.route('/plot-data')
def plot_data():
    """API endpoint to get plot data for the dashboard."""
    stats = get_dashboard_stats()
    
    if stats['analyzed'] > 0:
        return jsonify({
            'labels': ["No Misconceptions", "With Misconceptions"],
            'values': [stats['no_misconceptions'], stats['with_misconceptions']]
        })
    else:
        return jsonify({
            'labels': [],
            'values': []
        })

# Add scraper route to application
create_scraper_route(app)

# Add a nav link for the scraper in the layout
@app.context_processor
def inject_scraper_link():
    return dict(has_scraper=True)

if __name__ == '__main__':
    # Load misconception mapping and explanations on startup
    load_misconception_mapping()
    load_misconception_explanations()
    
    # Load the model if it exists
    model_file = os.path.join('models', 'model.joblib')
    if os.path.exists(model_file):
        print(f"Model marker file found at: {model_file}")
        try:
            # Try to load model components
            if os.path.exists(os.path.join('models', 'vectorizer.joblib')) and \
               os.path.exists(os.path.join('models', 'label_binarizer.joblib')) and \
               os.path.exists(os.path.join('models', 'classifier.joblib')):
                
                # Try to load the model
                print("Loading model from files...")
                model._load_model()
                model_trained = model.is_fitted
                print(f"Model loaded. Model is_fitted={model.is_fitted}")
                print("Model files found. Skipping automatic training.")
            else:
                print("Incomplete model files found. Will train model...")
                train_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will train model due to load error...")
            train_model()
    else:
        # Train the model automatically on startup
        print("No model found. Starting automatic model training...")
        train_model()
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)