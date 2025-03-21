import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
import os
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from sklearn.utils.class_weight import compute_class_weight
import json
import logging
import sys
from collections import Counter
import shutil
from datetime import datetime
from tqdm import tqdm

# CONFIGURATION
# Embedding models to test
EMBEDDING_MODELS = [
    'all-MiniLM-L6-v2',
    'sentence-t5-base'
]

# PCA configurations to test
PCA_CONFIGS = [
    {'use_pca': False, 'n_components': None},
    {'use_pca': True, 'n_components': 100},
    {'use_pca': True, 'n_components': 200}
]

# Logistic Regression parameters to test
LR_PARAM_GRID = {
    'C': [0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear', 'saga'],
    'class_weight': ['balanced']
}

# Combined configuration for all experiments
MODEL_CONFIGS = []

# Generate all combinations of parameters
for embedding_model in EMBEDDING_MODELS:
    for pca_config in PCA_CONFIGS:
        for c_value in LR_PARAM_GRID['C']:
            for solver in LR_PARAM_GRID['solver']:
                MODEL_CONFIGS.append({
                    'embedding_model': embedding_model,
                    'use_pca': pca_config['use_pca'],
                    'n_components': pca_config['n_components'],
                    'C': c_value,
                    'solver': solver,
                    'class_weight': 'balanced'
                })

# Print total number of configurations
print(f"Total configurations to test: {len(MODEL_CONFIGS)}")

# Output directories
OUTPUT_DIR = 'results_logistic_regression'
MODELS_DIR = f'{OUTPUT_DIR}/models'
CONFUSION_MATRICES_DIR = f'{OUTPUT_DIR}/confusion_matrices'
PREDICTIONS_DIR = f'{OUTPUT_DIR}/predictions'
METRICS_DIR = f'{OUTPUT_DIR}/metrics'


def setup_directories():
    """Create necessary output directories and clean existing results"""
    # Define all directories
    directories = [
        OUTPUT_DIR,
        MODELS_DIR,
        CONFUSION_MATRICES_DIR,
        PREDICTIONS_DIR,
        METRICS_DIR
    ]
    
    # Clean results directory if it exists
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning existing results in {OUTPUT_DIR}")
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
    
    # Create directories (or ensure they exist)
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"Created output directories in {OUTPUT_DIR}")


def load_data(data_path):
    """Load and prepare the initial datasets"""
    data_targets_name = 'keyword analysis - mapping.csv'
    data_texts_name = 'domains.csv'

    targets = pd.read_csv(os.path.join(data_path, data_targets_name))
    texts = pd.read_csv(os.path.join(data_path, data_texts_name))
    
    return targets, texts


def prepare_data(texts, targets):
    """Prepare and merge the data for training"""
    # Filter texts with non-null and non-empty descriptions
    texts = texts[texts['description'].notna() & (texts['description'] != '')]
    
    # Merge and prepare training data
    train_data = pd.merge(texts, targets, on='domain', how='right')
    train_data['category'] = np.where(
        train_data['category_x'].isna(),
        train_data['category_y'],
        train_data['category_x']
    )
    train_data = train_data.drop(['category_x', 'category_y'], axis=1)

    # Remove rows with NaN or empty descriptions
    train_data = train_data.dropna(subset=['description'])
    train_data = train_data[train_data['description'] != '']

    return train_data


def split_data(train_data):
    """Split data into training and test sets"""
    X = train_data['description']
    y = train_data['category']
    
    print("\nClass distribution before split:")
    print(y.value_counts())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    print("\nClass distribution in training set:")
    print(y_train.value_counts())
    print("\nClass distribution in test set:")
    print(y_test.value_counts())
    
    return X_train, X_test, y_train, y_test


def create_embeddings(X_train, X_test, model_name):
    """Create embeddings using specified SentenceTransformer model"""
    logging.info(f"Creating embeddings with model: {model_name}")
    embedding_model = SentenceTransformer(model_name)
    X_train_embeddings = embedding_model.encode(X_train.tolist(), show_progress_bar=True)
    X_test_embeddings = embedding_model.encode(X_test.tolist(), show_progress_bar=True)
    return embedding_model, X_train_embeddings, X_test_embeddings


def apply_pca(X_train_embeddings, X_test_embeddings, n_components):
    """Apply PCA dimensionality reduction"""
    logging.info(f"Applying PCA with {n_components} components")
    pca = PCA(n_components=n_components)
    X_train_reduced = pca.fit_transform(X_train_embeddings)
    X_test_reduced = pca.transform(X_test_embeddings)
    
    # Log explained variance
    explained_variance = np.sum(pca.explained_variance_ratio_)
    logging.info(f"PCA explained variance: {explained_variance:.4f}")
    
    return pca, X_train_reduced, X_test_reduced


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
    plt.savefig(f'{CONFUSION_MATRICES_DIR}/{filename}.png')
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
    plt.savefig(f'{CONFUSION_MATRICES_DIR}/{filename}_normalized.png')
    plt.close()


def train_model_with_grid_search(X_train, y_train):
    """Train logistic regression model with grid search for optimal parameters"""
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # Setup base model
    base_model = LogisticRegression(class_weight=class_weight_dict, max_iter=1000, random_state=42)
    
    # Setup grid search
    grid_search = GridSearchCV(
        base_model,
        param_grid=LR_PARAM_GRID,
        cv=5,
        scoring='recall_macro',  # Focus on recall
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    # Fit the grid search
    logging.info("Starting grid search for optimal logistic regression parameters...")
    grid_search.fit(X_train, y_train)
    
    # Log best parameters
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, class_weight_dict


def aggregate_predictions_by_domain(domains, y_true, y_pred, targets_df):
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
        
        # Get true category from targets dataset
        domain_true_category = targets_df[targets_df['domain'] == domain]['category'].iloc[0] if domain in targets_df['domain'].values else "unknown"
        
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


def save_domain_confusion_matrix(domain_df, title, model_id):
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
    plt.savefig(f'{CONFUSION_MATRICES_DIR}/domain_confusion_matrix_{model_id}.png')
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
    plt.savefig(f'{CONFUSION_MATRICES_DIR}/domain_confusion_matrix_{model_id}_normalized.png')
    plt.close()


def evaluate_model(clf, X_test, y_test, domains_test, targets_df, model_id, config):
    """Evaluate the model and save results"""
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Save URL-level results
    url_results_df = pd.DataFrame({
        'domain': domains_test,
        'true_category': y_test,
        'predicted_category': y_pred
    })
    url_results_df.to_csv(f'{PREDICTIONS_DIR}/prediction_{model_id}_url.csv', index=False)

    # Save URL-level confusion matrix
    save_confusion_matrix(
        y_test, y_pred, 
        f"Confusion Matrix ({model_id} - Test Set)",
        f"confusion_matrix_{model_id}_test"
    )
    
    # Generate classification report for URL-level predictions
    url_report = classification_report(y_test, y_pred, output_dict=True)
    url_report_df = pd.DataFrame(url_report).transpose()
    url_report_df.to_csv(f'{METRICS_DIR}/url_classification_report_{model_id}.csv')
    
    # Aggregate and evaluate domain-level predictions
    domain_df = aggregate_predictions_by_domain(domains_test, y_test, y_pred, targets_df)
    
    # Save domain-level predictions
    domain_df.to_csv(f'{PREDICTIONS_DIR}/prediction_{model_id}_domain.csv', index=False)
    
    # Save domain-level confusion matrix
    save_domain_confusion_matrix(
        domain_df,
        f"Confusion Matrix ({model_id} - Test Set)",
        model_id
    )
    
    # Generate classification report for domain-level predictions
    domain_report = classification_report(domain_df['true_category'], domain_df['predicted_category'], output_dict=True)
    domain_report_df = pd.DataFrame(domain_report).transpose()
    domain_report_df.to_csv(f'{METRICS_DIR}/domain_classification_report_{model_id}.csv')
    
    # Calculate partner and partner-seo recall
    y_true = domain_df['true_category']
    y_pred = domain_df['predicted_category']
    
    # Create confusion matrix for manual calculation
    labels = sorted(set(list(y_true.unique()) + list(y_pred.unique())))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    # Calculate recalls
    metrics = {}
    
    # For partner class
    if 'partner' in cm_df.index:
        partner_tp = cm_df.loc['partner', 'partner']
        partner_total = cm_df.loc['partner'].sum()
        partner_recall = partner_tp / partner_total if partner_total > 0 else 0
        metrics['partner_recall'] = partner_recall
    else:
        metrics['partner_recall'] = 0
    
    # For partner-seo class
    if 'partner-seo' in cm_df.index:
        seo_tp = cm_df.loc['partner-seo', 'partner-seo']
        seo_total = cm_df.loc['partner-seo'].sum()
        seo_recall = seo_tp / seo_total if seo_total > 0 else 0
        metrics['partner_seo_recall'] = seo_recall
    else:
        metrics['partner_seo_recall'] = 0
    
    # Calculate weighted score and accuracy
    metrics['weighted_score'] = (metrics.get('partner_recall', 0) * 0.7) + (metrics.get('partner_seo_recall', 0) * 0.3)
    metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)
    
    # Add configuration details
    metrics.update(config)
    
    return metrics


def save_model_artifacts(clf, config, embedding_model, pca=None):
    """Save model and related artifacts"""
    model_id = config['model_id']
    
    # Save the classifier model
    joblib.dump(clf, f'{MODELS_DIR}/classifier_{model_id}.joblib')
    
    # Save the embedding model
    embedding_model.save(f'{MODELS_DIR}/embedding_model_{model_id}')
    
    # Save PCA if used
    if pca is not None:
        joblib.dump(pca, f'{MODELS_DIR}/pca_{model_id}.joblib')
    
    # Save model configuration
    with open(f'{MODELS_DIR}/config_{model_id}.json', 'w') as f:
        json.dump(config, f, indent=4)


def rank_models():
    """Rank all models based on how well they predict 'partner' and 'partner-seo' classes"""
    logging.info("Ranking models based on domain-level metrics...")
    
    # Get all metrics files
    all_metrics = []
    for file in os.listdir(METRICS_DIR):
        if file.startswith('model_metrics_'):
            metrics_df = pd.read_csv(os.path.join(METRICS_DIR, file))
            all_metrics.append(metrics_df)
    
    if not all_metrics:
        logging.warning("No metrics files found to rank models")
        return
    
    # Combine all metrics
    combined_metrics = pd.concat(all_metrics)
    
    # Sort by weighted score (descending)
    ranked_models = combined_metrics.sort_values('weighted_score', ascending=False).reset_index(drop=True)
    
    # Save ranked models
    ranked_models.to_csv(f'{OUTPUT_DIR}/model_rankings.csv', index=False)
    
    # Log top models
    logging.info("\nTop 5 models by weighted score:")
    top_models = ranked_models.head(5)
    for idx, row in top_models.iterrows():
        logging.info(f"Rank {idx+1}: {row['model_id']} - Weighted Score: {row['weighted_score']:.4f}")
        logging.info(f"  Partner Recall: {row['partner_recall']:.4f}, Partner-SEO Recall: {row['partner_seo_recall']:.4f}")
        logging.info(f"  Config: Embedding={row['embedding_model']}, PCA={row['pca_info']}, C={row['C']}, Solver={row['solver']}")
    
    return ranked_models


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{OUTPUT_DIR}/training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main execution function"""
    try:
        # Setup logging and directories
        setup_directories()
        setup_logging()
        
        logging.info("Starting logistic regression grid search experiments")
        logging.info(f"Total configurations to test: {len(MODEL_CONFIGS)}")
        
        # Load and prepare data
        logging.info("Loading data...")
        targets, texts = load_data("data")
        
        train_data = prepare_data(texts, targets)
        logging.info(f"Prepared data shape: {train_data.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(train_data)
        
        # Get domains for test set
        domains_test = train_data.loc[X_test.index, 'domain']
        
        # List to store all model metrics
        all_model_metrics = []
    
        # Counter for model IDs
        model_counter = 0
        
        # Dictionary to cache embedding models
        embedding_models_cache = {}
        
        # Loop through all configurations
        for config in tqdm(MODEL_CONFIGS, desc="Testing model configurations"):
            embedding_model_name = config['embedding_model']
            
            # Create or retrieve embeddings from cache
            if embedding_model_name not in embedding_models_cache:
                logging.info(f"Creating embeddings with model: {embedding_model_name}")
                embedding_model, X_train_embeddings, X_test_embeddings = create_embeddings(
                    X_train, X_test, embedding_model_name
                )
                embedding_models_cache[embedding_model_name] = (embedding_model, X_train_embeddings, X_test_embeddings)
            else:
                logging.info(f"Using cached embeddings for model: {embedding_model_name}")
                embedding_model, X_train_embeddings, X_test_embeddings = embedding_models_cache[embedding_model_name]
            
            # Apply PCA if needed
            if config['use_pca']:
                n_components = config['n_components']
                pca_info = f"pca{n_components}"
                pca, X_train_processed, X_test_processed = apply_pca(
                    X_train_embeddings, X_test_embeddings, n_components
                )
            else:
                pca = None
                X_train_processed = X_train_embeddings
                X_test_processed = X_test_embeddings
                pca_info = "nopca"
            
            # Create descriptive prefix like in train_model.py
            embedding_prefix = embedding_model_name.replace('all-', '').replace('-v2', '')
            prefix = f"lr_{embedding_prefix}_{pca_info}_C{config['C']}_{config['solver']}"
            
            # Use prefix as model ID
            model_id = prefix
            model_counter += 1
            
            # Update config with model ID and PCA info
            model_config = config.copy()
            model_config['model_id'] = model_id
            model_config['pca_info'] = pca_info
            
            # Train Logistic Regression model
            logging.info(f"Training model {model_id}: Embedding={embedding_model_name}, PCA={pca_info}, C={config['C']}, Solver={config['solver']}")
            
            # Calculate class weights
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = dict(zip(np.unique(y_train), class_weights))
            
            # Create and fit model
            clf = LogisticRegression(
                C=config['C'],
                solver=config['solver'],
                class_weight=class_weight_dict,
                max_iter=1000,
                random_state=42
            )
            clf.fit(X_train_processed, y_train)
            
            # Evaluate model
            logging.info(f"Evaluating model {model_id}...")
            metrics = evaluate_model(
                clf, X_test_processed, y_test, domains_test, targets, model_id, model_config
            )
            
            # Save model artifacts
            save_model_artifacts(clf, model_config, embedding_model, pca)
            
            # Add metrics to list
            all_model_metrics.append(metrics)
            
            # Save metrics for this model
            pd.DataFrame([metrics]).to_csv(f'{METRICS_DIR}/model_metrics_{model_id}.csv', index=False)
            
            # Log interim results
            logging.info(f"Model {model_id} - Weighted Score: {metrics['weighted_score']:.4f}")
            logging.info(f"  Partner Recall: {metrics['partner_recall']:.4f}, "
                       f"Partner-SEO Recall: {metrics['partner_seo_recall']:.4f}")
        
        # Create combined metrics file
        all_metrics_df = pd.DataFrame(all_model_metrics)
        all_metrics_df.to_csv(f'{OUTPUT_DIR}/all_model_metrics.csv', index=False)
        
        # Rank models and save rankings
        ranked_models = rank_models()
        
        # Log completion
        logging.info(f"Experiments completed. Trained and evaluated {model_counter} models.")
        if ranked_models is not None and not ranked_models.empty:
            top_model = ranked_models.iloc[0]
            logging.info(f"Best model: {top_model['model_id']} - Weighted Score: {top_model['weighted_score']:.4f}")
            logging.info(f"Config: Embedding={top_model['embedding_model']}, PCA={top_model['pca_info']}, C={top_model['C']}, Solver={top_model['solver']}")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
