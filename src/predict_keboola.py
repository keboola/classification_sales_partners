import pandas as pd
import numpy as np
import os
import sys
import json
import logging
import joblib
from sentence_transformers import SentenceTransformer
from collections import Counter
from pathlib import Path

# Keboola folders with updated file paths
INPUT_PATH = '/data/in/tables/'
OUTPUT_PATH = '/data/out/tables/'
MODEL_PATH = '/data/in/files/'  # Model is input for prediction
MODEL_NAME = 'best_classifier_model.joblib'
EMBEDDING_MODEL_NAME = 'embedding_model'

# Input and output file names
INPUT_FILE = 'data_domains_classification.csv'
OUTPUT_FILE = 'data_domains_predictions.csv'

def setup_directories():
    """Create necessary output directories for Keboola"""
    # Ensure output path exists
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

def setup_logging():
    """Setup logging configuration for Keboola"""
    log_file = os.path.join(OUTPUT_PATH, 'prediction_log.txt')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Log system information
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(f"Input path exists: {os.path.exists(INPUT_PATH)}")
    logging.info(f"Model path exists: {os.path.exists(MODEL_PATH)}")
    logging.info(f"Output path exists: {os.path.exists(OUTPUT_PATH)}")

def load_model():
    """Load the trained model and embedding model from Keboola storage"""
    # Load classifier
    model_file = os.path.join(MODEL_PATH, MODEL_NAME)
    if not os.path.exists(model_file):
        logging.error(f"Model file not found: {model_file}")
        list_dir = os.listdir(MODEL_PATH) if os.path.exists(MODEL_PATH) else []
        logging.error(f"Available files in model path: {list_dir}")
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    # Load the classifier
    logging.info(f"Loading classifier from {model_file}")
    clf = joblib.load(model_file)
    
    # Load embedding model
    embedding_model_dir = os.path.join(MODEL_PATH, EMBEDDING_MODEL_NAME)
    if not os.path.exists(embedding_model_dir):
        logging.error(f"Embedding model directory not found: {embedding_model_dir}")
        raise FileNotFoundError(f"Embedding model directory not found: {embedding_model_dir}")
    
    logging.info(f"Loading embedding model from {embedding_model_dir}")
    embedding_model = SentenceTransformer(embedding_model_dir)
    
    return clf, embedding_model

def load_domains():
    """Load domains data from Keboola input tables"""
    # Updated domains file path
    domains_file = os.path.join(INPUT_PATH, INPUT_FILE)
    
    if not os.path.exists(domains_file):
        logging.error(f"Domains file not found: {domains_file}")
        list_dir = os.listdir(INPUT_PATH) if os.path.exists(INPUT_PATH) else []
        logging.error(f"Available files in input: {list_dir}")
        raise FileNotFoundError(f"Domains file not found: {domains_file}")
    
    # Log file size for debugging
    logging.info(f"Domains file size: {os.path.getsize(domains_file)} bytes")
    
    # Load dataset
    texts = pd.read_csv(domains_file)
    logging.info(f"Loaded domains data with shape: {texts.shape}")
    
    # Check for required columns
    if 'domain' not in texts.columns or 'description' not in texts.columns:
        missing_cols = []
        if 'domain' not in texts.columns:
            missing_cols.append('domain')
        if 'description' not in texts.columns:
            missing_cols.append('description')
        raise ValueError(f"Missing required columns in domains data: {missing_cols}")
    
    return texts

def aggregate_predictions_by_domain(domains, predictions):
    """
    Aggregate predictions by domain following the rules:
    1. If any prediction for a domain is 'partner', the domain is 'partner'
    2. If any prediction is 'partner-seo' and none are 'partner', the domain is 'partner-seo'
    3. Otherwise, use the majority prediction
    """
    # Create DataFrame with domains and predictions
    df = pd.DataFrame({
        'domain': domains,
        'predicted_category': predictions
    })
    
    # Group by domain to aggregate results
    domain_results = []
    
    for domain, group in df.groupby('domain'):
        # Get all predictions for this domain
        predictions = group['predicted_category'].tolist()
        
        # Count occurrences of each category
        prediction_counts = Counter(predictions)
        total_urls = len(predictions)
        
        # Determine aggregated prediction based on rules
        if 'partner' in prediction_counts:
            agg_prediction = 'partner'
        elif 'partner-seo' in prediction_counts:
            agg_prediction = 'partner-seo'
        else:
            # Use most common prediction
            agg_prediction = prediction_counts.most_common(1)[0][0]
        
        # Calculate counts and percentages
        partner_count = prediction_counts.get('partner', 0)
        partner_seo_count = prediction_counts.get('partner-seo', 0)
        
        partner_percentage = (partner_count / total_urls) * 100 if total_urls > 0 else 0
        partner_seo_percentage = (partner_seo_count / total_urls) * 100 if total_urls > 0 else 0
        
        domain_results.append({
            'domain': domain,
            'predicted_category': agg_prediction,
            'partner_count': partner_count,
            'partner_seo_count': partner_seo_count,
            'total_urls': total_urls,
            'partner_percentage': partner_percentage,
            'partner_seo_percentage': partner_seo_percentage
        })
    
    return pd.DataFrame(domain_results)

def make_predictions(clf, embedding_model, texts):
    """Make predictions on domains data and save to Keboola output"""
    # Filter texts with non-null and non-empty descriptions
    texts_filtered = texts[texts['description'].notna() & (texts['description'] != '')]
    logging.info(f"Making predictions on {len(texts_filtered)} domains with descriptions")
    
    # Create embeddings for filtered texts
    logging.info("Creating embeddings...")
    texts_embeddings = embedding_model.encode(texts_filtered['description'].tolist(), show_progress_bar=True)
    
    # Make predictions
    logging.info("Making predictions...")
    predictions = clf.predict(texts_embeddings)
    
    # Add predictions to the dataframe
    texts_filtered['predicted_category'] = predictions
    
    # Save URL-level predictions
    url_results_df = texts_filtered[['domain', 'description', 'predicted_category']]
    url_results_path = os.path.join(OUTPUT_PATH, 'url_level_predictions.csv')
    url_results_df.to_csv(url_results_path, index=False)
    logging.info(f"Saved URL-level predictions to {url_results_path}")
    
    # Generate domain-level predictions
    domain_df = aggregate_predictions_by_domain(
        texts_filtered['domain'], 
        texts_filtered['predicted_category']
    )
    
    # Save domain-level predictions to specified output path
    domain_pred_path = os.path.join(OUTPUT_PATH, OUTPUT_FILE)
    domain_df.to_csv(domain_pred_path, index=False)
    logging.info(f"Saved domain-level predictions to {domain_pred_path}")
    
    # Create a summary of predictions by category
    prediction_summary = domain_df['predicted_category'].value_counts().reset_index()
    prediction_summary.columns = ['category', 'count']
    prediction_summary['percentage'] = (prediction_summary['count'] / len(domain_df) * 100).round(2)
    summary_path = os.path.join(OUTPUT_PATH, 'domains_predictions_summary.csv')
    prediction_summary.to_csv(summary_path, index=False)
    logging.info(f"Saved prediction summary to {summary_path}")
    
    # Log summary
    logging.info(f"Prediction distribution:\n{prediction_summary.to_string(index=False)}")
    
    return domain_df

def create_manifest_files():
    """Create manifest files for Keboola output tables"""
    for file in os.listdir(OUTPUT_PATH):
        if file.endswith('.csv'):
            manifest_path = os.path.join(OUTPUT_PATH, f"{file}.manifest")
            with open(manifest_path, 'w') as manifest_file:
                json.dump({
                    "primary_key": ["domain"] if "domain" in file else [],
                    "incremental": False,
                    "delimiter": ",",
                    "enclosure": "\"",
                    "columns": []  # Columns will be automatically detected
                }, manifest_file)
            logging.info(f"Created manifest for {file}")

def main():
    """Main execution function for prediction with Keboola"""
    try:
        # Setup directories
        setup_directories()
        
        # Setup logging
        setup_logging()
        logging.info("Starting prediction process with Keboola")
        
        # List input directory contents
        input_files = os.listdir(INPUT_PATH) if os.path.exists(INPUT_PATH) else []
        logging.info(f"Files in input directory: {input_files}")
        
        model_files = os.listdir(MODEL_PATH) if os.path.exists(MODEL_PATH) else []
        logging.info(f"Files in model directory: {model_files}")
        
        # Load the model and embedding model
        clf, embedding_model = load_model()
        logging.info("Models loaded successfully")
        
        # Load domains data
        texts = load_domains()
        
        # Make predictions
        domain_results = make_predictions(clf, embedding_model, texts)
        
        # Create manifest files for output tables
        create_manifest_files()
        
        logging.info("Prediction process completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}", exc_info=True)
        # Ensure error is visible in Keboola
        with open(os.path.join(OUTPUT_PATH, 'error.txt'), 'w') as f:
            f.write(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
