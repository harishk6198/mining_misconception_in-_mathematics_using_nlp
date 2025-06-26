import json
import os
import logging

logger = logging.getLogger(__name__)

def save_results(results):
    """Save analysis results to file."""
    try:
        output_dir = 'results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Convert numpy types to Python native types
        processed_results = []
        for result in results:
            try:
                # Handle question_id conversion
                if hasattr(result['question_id'], 'item'):
                    question_id = result['question_id'].item()
                else:
                    question_id = int(result['question_id'])
                
                # Handle misconception_ids conversion
                misconception_ids = []
                for mid in result['misconception_ids']:
                    try:
                        if hasattr(mid, 'item'):  # numpy type
                            misconception_ids.append(mid.item())
                        else:
                            misconception_ids.append(int(mid))
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert misconception ID: {mid}, skipping")
                
                # Extract misconception names from details
                misconception_names = []
                if 'misconception_details' in result:
                    for detail in result['misconception_details']:
                        misconception_names.append(detail['name'])
                
                processed_result = {
                    'question_id': question_id,
                    'question_text': result.get('question_text', ''),
                    'misconception_ids': misconception_ids,
                    'misconception_names': misconception_names,
                    'misconception_details': result.get('misconception_details', [])
                }
                processed_results.append(processed_result)
            except Exception as item_error:
                logger.error(f"Error processing result item: {str(item_error)}")
                continue

        output_file = os.path.join(output_dir, 'misconceptions.json')
        with open(output_file, 'w') as f:
            json.dump(processed_results, f, indent=4)
        
        logger.info(f"Saved {len(processed_results)} results to {output_file}")
        return True

    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        return False
