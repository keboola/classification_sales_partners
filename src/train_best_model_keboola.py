import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
import logging
from collections import Counter
import random
import torch
from pathlib import Path

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Keboola folders with updated file paths
INPUT_PATH = '/data/in/tables/'
OUTPUT_PATH = '/data/out/tables/'
MODEL_PATH = '/data/out/files/'
MODEL_NAME = 'best_classifier_model.joblib'
EMBEDDING_MODEL_NAME = 'embedding_model'

# Input file names
DOMAINS_TRAIN_FILE = 'domains_train.csv'

# Output file name
OUTPUT_FILE = 'data_domains_predictions.csv'

# MODEL PARAMETERS
MODEL_NAME_TRANSFORMER = 'all-MiniLM-L6-v2'

# Logistic Regression parameters
LOGISTIC_REGRESSION_PARAMS = {
    'C': 1.0,
    'max_iter': 1000,
    'random_state': 42,
    'solver': 'lbfgs',
    'multi_class': 'auto'
}

def setup_directories():
    """Create necessary output directories for Keboola"""
    # Create model output directory
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
    
    # Create confusion matrix directory (within model path for Keboola)
    Path(os.path.join(MODEL_PATH, 'confusion_matrices')).mkdir(parents=True, exist_ok=True)
    
    # Ensure output path exists
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

def load_data():
    """Load the data files from Keboola input tables"""
    # Updated file paths
    domains_train_file = os.path.join(INPUT_PATH, DOMAINS_TRAIN_FILE)
    
    if not os.path.exists(domains_train_file):
        logging.error(f"Training file not found: {domains_train_file}")
        list_dir = os.listdir(INPUT_PATH)
        logging.error(f"Available files in input: {list_dir}")
        raise FileNotFoundError(f"Training file not found: {domains_train_file}")
    
    # Log file sizes for debugging
    logging.info(f"Training file size: {os.path.getsize(domains_train_file)} bytes")

    # Load dataset
    train_data = pd.read_csv(domains_train_file)
    logging.info(f"Loaded training data with shape: {train_data.shape}")
    
    # Check if required columns exist
    required_columns = ['domain', 'description', 'category']
    missing_columns = [col for col in required_columns if col not in train_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in training data: {missing_columns}")
    
    return train_data

def prepare_data(data):
    """Prepare data for training"""
    # Filter data with non-null and non-empty descriptions
    data = data[data['description'].notna() & (data['description'] != '')]
    
    # Remove rows with NaN categories
    data = data.dropna(subset=['category'])
    
    return data

def create_embeddings(X):
    """Create embeddings using SentenceTransformer"""
    embedding_model = SentenceTransformer(MODEL_NAME_TRANSFORMER)
    X_embeddings = embedding_model.encode(X.tolist(), show_progress_bar=True)
    return embedding_model, X_embeddings

def train_model(X_embeddings, y):
    """Train logistic regression with class weights"""
    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weight_dict = dict(zip(np.unique(y), class_weights))
    logging.info(f"Computed class weights: {class_weight_dict}")
    
    # Train Logistic Regression model with class weights
    clf = LogisticRegression(
        **LOGISTIC_REGRESSION_PARAMS,
        class_weight=class_weight_dict
    )
    clf.fit(X_embeddings, y)
    
    return clf, class_weight_dict

def save_model_artifacts(clf, class_weight_dict, embedding_model):
    """Save model and related artifacts to Keboola storage"""
    # Save the model
    model_file = os.path.join(MODEL_PATH, MODEL_NAME)
    joblib.dump(clf, model_file)
    logging.info(f"Saved model to {model_file}")
    
    # Save the embedding model
    embedding_model_dir = os.path.join(MODEL_PATH, EMBEDDING_MODEL_NAME)
    embedding_model.save(embedding_model_dir)
    logging.info(f"Saved embedding model to {embedding_model_dir}")
    
    # Save class weights
    weights_file = os.path.join(MODEL_PATH, 'class_weights.json')
    with open(weights_file, 'w') as f:
        json.dump(class_weight_dict, f)
    logging.info(f"Saved class weights to {weights_file}")
    
    # Save model config
    config_file = os.path.join(MODEL_PATH, 'model_config.json')
    with open(config_file, 'w') as f:
        json.dump({
            'model_type': 'LogisticRegression',
            'parameters': LOGISTIC_REGRESSION_PARAMS,
            'embedding_model': MODEL_NAME_TRANSFORMER,
            'class_weights_used': True,
            'trained_on_full_dataset': True
        }, f, indent=4)
    logging.info(f"Saved model config to {config_file}")

def aggregate_predictions_by_domain(domains, y_true, y_pred, data_df):
    """
    Aggregate predictions by domain following the rules:
    1. If any prediction for a domain is 'partner', the domain is 'partner'
    2. If any prediction is 'partner-seo' and none are 'partner', the domain is 'partner-seo'
    3. Otherwise, use the majority prediction
    """
    # Create DataFrame with domains, true labels, and predictions
    df = pd.DataFrame({
        'domain': domains,
        'url_true_category': y_true,
        'predicted_category': y_pred
    })
    
    # Group by domain to aggregate results
    domain_results = []
    
    for domain, group in df.groupby('domain'):
        # Get all predictions for this domain
        predictions = group['predicted_category'].tolist()
        
        # Count occurrences of each category
        prediction_counts = Counter(predictions)
        total_urls = len(predictions)
        
        # Get true category from data_df dataset
        domain_true_category = data_df[data_df['domain'] == domain]['category'].iloc[0] if domain in data_df['domain'].values else "unknown"
        
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
            'true_category': domain_true_category,
            'predicted_category': agg_prediction,
            'partner_count': partner_count,
            'partner_seo_count': partner_seo_count,
            'total_urls': total_urls,
            'partner_percentage': partner_percentage,
            'partner_seo_percentage': partner_seo_percentage
        })
    
    return pd.DataFrame(domain_results)

def save_confusion_matrix(y_true, y_pred, title, filename):
    """Create and save both regular and normalized confusion matrices"""
    # Regular confusion matrix
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=sorted(pd.Series(y_true).unique()),
        yticklabels=sorted(pd.Series(y_true).unique())
    )
    plt.title(f'{title}')
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, 'confusion_matrices', f'{filename}.png'))
    plt.close()

    # Normalized confusion matrix
    plt.figure(figsize=(12, 8))
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=sorted(pd.Series(y_true).unique()),
        yticklabels=sorted(pd.Series(y_true).unique())
    )

    plt.title(f'Normalized {title}')
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, 'confusion_matrices', f'{filename}_normalized.png'))
    plt.close()

def save_domain_confusion_matrix(domain_df, title):
    """Create and save confusion matrix for domain-level predictions"""
    y_true = domain_df['true_category']
    y_pred = domain_df['predicted_category']
    
    # Regular confusion matrix
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=sorted(pd.Series(y_true).unique()),
        yticklabels=sorted(pd.Series(y_true).unique())
    )
    plt.title(f'{title} (Domain Level)')
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, 'confusion_matrices', 'domain_confusion_matrix.png'))
    plt.close()

    # Normalized confusion matrix
    plt.figure(figsize=(12, 8))
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=sorted(pd.Series(y_true).unique()),
        yticklabels=sorted(pd.Series(y_true).unique())
    )

    plt.title(f'Normalized {title} (Domain Level)')
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, 'confusion_matrices', 'domain_confusion_matrix_normalized.png'))
    plt.close()

def evaluate_model(clf, X_embeddings, y, domains, data_df):
    """Evaluate the model and save results to Keboola output tables"""
    # Make predictions
    y_pred = clf.predict(X_embeddings)
    
    # Save URL-level results
    url_results_df = pd.DataFrame({
        'domain': domains,
        'true_category': y,
        'predicted_category': y_pred
    })
    url_results_path = os.path.join(OUTPUT_PATH, 'url_level_predictions.csv')
    url_results_df.to_csv(url_results_path, index=False)
    logging.info(f"Saved URL-level predictions to {url_results_path}")
    
    # Save URL-level confusion matrix
    save_confusion_matrix(
        y, y_pred, 
        "URL-Level Confusion Matrix (LR with Class Weights - Full Dataset)",
        "url_confusion_matrix"
    )
    
    # Aggregate and evaluate domain-level predictions
    domain_df = aggregate_predictions_by_domain(domains, y, y_pred, data_df)
    
    # Save domain-level predictions
    domain_pred_path = os.path.join(OUTPUT_PATH, OUTPUT_FILE)
    domain_df.to_csv(domain_pred_path, index=False)
    logging.info(f"Saved domain-level predictions to {domain_pred_path}")
    
    # Save domain-level confusion matrix
    save_domain_confusion_matrix(
        domain_df,
        "Domain-Level Confusion Matrix (LR with Class Weights - Full Dataset)"
    )
    
    # Calculate partner and partner-seo recall
    y_true = domain_df['true_category']
    y_pred = domain_df['predicted_category']
    
    # Create confusion matrix for manual calculation
    labels = sorted(set(list(y_true.unique()) + list(y_pred.unique())))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    # Calculate and log recalls
    metrics = {}
    
    # For partner class
    if 'partner' in cm_df.index:
        partner_tp = cm_df.loc['partner', 'partner']
        partner_total = cm_df.loc['partner'].sum()
        partner_recall = partner_tp / partner_total if partner_total > 0 else 0
        metrics['partner_recall'] = partner_recall
        logging.info(f"\nPartner class: {partner_tp} correctly classified out of {partner_total} total")
        
        if partner_total > 0 and partner_tp < partner_total:
            partner_errors = cm_df.loc['partner'].drop('partner')
            for idx, val in partner_errors.items():
                if val > 0:
                    logging.info(f"  {val} 'partner' samples misclassified as '{idx}'")
    
    # For partner-seo class
    if 'partner-seo' in cm_df.index:
        seo_tp = cm_df.loc['partner-seo', 'partner-seo']
        seo_total = cm_df.loc['partner-seo'].sum()
        seo_recall = seo_tp / seo_total if seo_total > 0 else 0
        metrics['partner_seo_recall'] = seo_recall
        logging.info(f"\nPartner-SEO class: {seo_tp} correctly classified out of {seo_total} total")
        
        if seo_total > 0 and seo_tp < seo_total:
            seo_errors = cm_df.loc['partner-seo'].drop('partner-seo')
            for idx, val in seo_errors.items():
                if val > 0:
                    logging.info(f"  {val} 'partner-seo' samples misclassified as '{idx}'")
    
    # Calculate weighted score (prioritizing recall of important classes)
    metrics['weighted_score'] = (metrics.get('partner_recall', 0) * 0.7) + (metrics.get('partner_seo_recall', 0) * 0.3)
    metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)
    
    # Save metrics to Keboola output
    metrics_path = os.path.join(OUTPUT_PATH, 'key_metrics.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    logging.info(f"Saved key metrics to {metrics_path}")
    
    # Log final results
    logging.info("\nModel evaluation complete (on full dataset)")
    logging.info(f"Partner class recall: {metrics.get('partner_recall', 0):.4f}")
    logging.info(f"Partner-SEO class recall: {metrics.get('partner_seo_recall', 0):.4f}")
    logging.info(f"Weighted score: {metrics.get('weighted_score', 0):.4f}")
    logging.info(f"Overall accuracy: {metrics.get('overall_accuracy', 0):.4f}")
    
    return y_pred, metrics

def predict_on_all_domains(data, clf, embedding_model):
    """Make predictions on all domains with descriptions and save to Keboola output"""
    # Filter data with non-null and non-empty descriptions
    data_filtered = data[data['description'].notna() & (data['description'] != '')]
    
    # Create embeddings for filtered texts
    logging.info(f"Creating embeddings for {len(data_filtered)} domains...")
    data_embeddings = embedding_model.encode(data_filtered['description'].tolist(), show_progress_bar=True)
    
    # Make predictions
    logging.info("Making predictions on all domains...")
    predictions = clf.predict(data_embeddings)
    
    # Add predictions to the dataframe
    data_filtered['predicted_category'] = predictions
    
    # Save to CSV in Keboola output
    output_df = data_filtered[['domain', 'description', 'predicted_category']]
    domains_pred_path = os.path.join(OUTPUT_PATH, OUTPUT_FILE)
    output_df.to_csv(domains_pred_path, index=False)
    
    # Create a summary of predictions by category
    prediction_summary = output_df['predicted_category'].value_counts().reset_index()
    prediction_summary.columns = ['category', 'count']
    prediction_summary['percentage'] = (prediction_summary['count'] / len(output_df) * 100).round(2)
    summary_path = os.path.join(OUTPUT_PATH, 'domains_predictions_summary.csv')
    prediction_summary.to_csv(summary_path, index=False)
    
    logging.info(f"Saved predictions for {len(output_df)} domains to {domains_pred_path}")
    logging.info(f"Saved prediction summary to {summary_path}")
    logging.info(f"Prediction distribution:\n{prediction_summary.to_string(index=False)}")

def setup_logging():
    """Setup logging configuration for Keboola"""
    log_file = os.path.join(MODEL_PATH, 'training_log.log')
    
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
    logging.info(f"Output path exists: {os.path.exists(OUTPUT_PATH)}")
    logging.info(f"Model path exists: {os.path.exists(MODEL_PATH)}")

def main():
    """Main execution function for Keboola integration"""
    try:
        # Setup directories first
        setup_directories()
        
        # Setup logging after directories are created
        setup_logging()
        logging.info("Starting model training process on full dataset for Keboola")
        
        # List input directory contents
        input_files = os.listdir(INPUT_PATH) if os.path.exists(INPUT_PATH) else []
        logging.info(f"Files in input directory: {input_files}")
        
        # Load data from Keboola
        logging.info("Loading data from Keboola...")
        train_data = load_data()
        
        # Prepare data
        data = prepare_data(train_data)
        logging.info(f"Prepared data shape: {data.shape}")
        
        # Print class distribution
        logging.info("\nClass distribution in full dataset:")
        class_distribution = data['category'].value_counts()
        logging.info(f"\n{class_distribution}")
        
        # Get features and labels from the full dataset
        X = data['description']
        y = data['category']
        domains = data['domain']
        
        # Create embeddings
        logging.info("Creating embeddings...")
        embedding_model, X_embeddings = create_embeddings(X)
        
        # Train the model on full dataset
        logging.info("Training logistic regression with class weights on full dataset...")
        clf, class_weight_dict = train_model(X_embeddings, y)
        
        # Evaluate the model on the same full dataset
        logging.info("Evaluating model on full dataset...")
        y_pred, metrics = evaluate_model(clf, X_embeddings, y, domains, data)

        # Save model and artifacts to Keboola storage
        logging.info("Saving model and artifacts to Keboola storage...")
        save_model_artifacts(clf, class_weight_dict, embedding_model)

        # Create manifest files for output tables
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

        logging.info("Training and prediction on full dataset completed successfully!")

    except Exception as e:
        logging.error(f"Error during training or prediction: {str(e)}", exc_info=True)
        # Ensure error is visible in Keboola
        with open(os.path.join(MODEL_PATH, 'error.txt'), 'w') as f:
            f.write(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
