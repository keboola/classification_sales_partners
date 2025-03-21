import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import json
import logging
import sys
from xgboost import XGBClassifier
import umap
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import shutil


# MODEL PARAMETERS
# Embedding model
MODEL_NAME_TRANSFORMER = 'all-MiniLM-L6-v2'

# UMAP
N_COMPONENTS_UMAP = 100

# Random Forest
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}

# XGBoost
XGB_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'random_state': 42
}

# Logistic Regression
LOGISTIC_REGRESSION_PARAMS = {
    'C': 1.0,
    'max_iter': 1000,
    'random_state': 42,
    'solver': 'lbfgs',
    'multi_class': 'auto'
}

# Configurations to test
MODEL_TYPES = ['rf', 'xgb', 'lr']  # Random Forest, XGBoost, Logistic Regression
USE_UMAP = [True, False]
USE_SMOTE = [True, False]
USE_CLASS_WEIGHTS = [True, False]

# Only testing combinations with UMAP=True to avoid too many models
MODEL_CONFIGS = [
    # With UMAP
    {"model_type": "rf", "use_umap": True, "use_smote": False, "use_class_weights": False},
    {"model_type": "rf", "use_umap": True, "use_smote": True, "use_class_weights": False},
    {"model_type": "rf", "use_umap": True, "use_smote": False, "use_class_weights": True},
    {"model_type": "rf", "use_umap": True, "use_smote": True, "use_class_weights": True},
    {"model_type": "xgb", "use_umap": True, "use_smote": False, "use_class_weights": False},
    {"model_type": "xgb", "use_umap": True, "use_smote": True, "use_class_weights": False},
    {"model_type": "xgb", "use_umap": True, "use_smote": False, "use_class_weights": True},
    {"model_type": "xgb", "use_umap": True, "use_smote": True, "use_class_weights": True},
    {"model_type": "lr", "use_umap": True, "use_smote": False, "use_class_weights": False},
    {"model_type": "lr", "use_umap": True, "use_smote": True, "use_class_weights": False},
    {"model_type": "lr", "use_umap": True, "use_smote": False, "use_class_weights": True},
    {"model_type": "lr", "use_umap": True, "use_smote": True, "use_class_weights": True},

    # Without UMAP - All combinations
    {"model_type": "rf", "use_umap": False, "use_smote": False, "use_class_weights": False},
    {"model_type": "rf", "use_umap": False, "use_smote": True, "use_class_weights": False},
    {"model_type": "rf", "use_umap": False, "use_smote": False, "use_class_weights": True},
    {"model_type": "rf", "use_umap": False, "use_smote": True, "use_class_weights": True},
    {"model_type": "xgb", "use_umap": False, "use_smote": False, "use_class_weights": False},
    {"model_type": "xgb", "use_umap": False, "use_smote": True, "use_class_weights": False},
    {"model_type": "xgb", "use_umap": False, "use_smote": False, "use_class_weights": True},
    {"model_type": "xgb", "use_umap": False, "use_smote": True, "use_class_weights": True},
    {"model_type": "lr", "use_umap": False, "use_smote": False, "use_class_weights": False},
    {"model_type": "lr", "use_umap": False, "use_smote": True, "use_class_weights": False},
    {"model_type": "lr", "use_umap": False, "use_smote": False, "use_class_weights": True},
    {"model_type": "lr", "use_umap": False, "use_smote": True, "use_class_weights": True},
]


def setup_directories():
    """Create necessary output directories and clean existing results"""
    # Define all directories
    directories = [
        'results', 
        'embeddings', 
        'models',
        'results/URLs',
        'results/Domains',
        'results/output_predictions'
    ]
    
    # Clean results directories if they exist
    results_dirs = [d for d in directories if d.startswith('results')]
    for directory in results_dirs:
        if os.path.exists(directory):
            # Delete all files in the directory but keep the directory itself
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logging.warning(f"Failed to delete {file_path}. Reason: {e}")
    
    # Create directories (or ensure they exist)
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def load_data(data_path):
    """Load and prepare the initial datasets"""
    data_targets_name = 'keyword analysis - mapping.csv'
    data_features_name = 'keyword analysis - domain ad info.csv'
    data_texts_name = 'domains.csv'

    targets = pd.read_csv(os.path.join(data_path, data_targets_name))
    features = pd.read_csv(os.path.join(data_path, data_features_name))
    texts = pd.read_csv(os.path.join(data_path, data_texts_name))
    
    return targets, features, texts


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


def create_embeddings(X_train, X_test):
    """Create embeddings using SentenceTransformer"""
    embedding_model = SentenceTransformer(MODEL_NAME_TRANSFORMER)
    X_train_embeddings = embedding_model.encode(X_train.tolist(), show_progress_bar=True)
    X_test_embeddings = embedding_model.encode(X_test.tolist(), show_progress_bar=True)
    return embedding_model, X_train_embeddings, X_test_embeddings


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
    plt.savefig(f'results/URLs/{filename}.png')
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
    plt.savefig(f'results/URLs/{filename}_normalized.png')
    plt.close()


def train_model(X_train, y_train, config):
    """Train model based on configuration"""
    # Apply SMOTE if specified
    if config["use_smote"]:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logging.info("Applied SMOTE. Class distribution after balancing:")
        for cls, count in zip(*np.unique(y_train, return_counts=True)):
            logging.info(f"  {cls}: {count}")
    
    # Calculate class weights if needed
    class_weight_dict = None
    if config["use_class_weights"]:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        logging.info(f"Computed class weights: {class_weight_dict}")
    
    # Initialize model based on type
    model_type = config["model_type"]
    
    if model_type == "rf":
        # Random Forest
        clf = RandomForestClassifier(
            **RANDOM_FOREST_PARAMS,
            class_weight=class_weight_dict
        )
        clf.fit(X_train, y_train)

    elif model_type == "xgb":
        # XGBoost requires numerical labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        
        # XGBoost
        if len(np.unique(y_train)) > 2:
            xgb_params = XGB_PARAMS.copy()
            xgb_params['objective'] = 'multi:softprob'
            xgb_params['num_class'] = len(np.unique(y_train_encoded))
            clf = XGBClassifier(**xgb_params)
            
            if class_weight_dict:
                # For multiclass, create sample weights
                sample_weights = np.ones(len(y_train_encoded))
                for i, cls in enumerate(label_encoder.classes_):
                    if cls in class_weight_dict:
                        sample_weights[y_train_encoded == i] = class_weight_dict[cls]
                clf.fit(X_train, y_train_encoded, sample_weight=sample_weights)
            else:
                clf.fit(X_train, y_train_encoded)     
        else:
            # Binary classification
            xgb_params = XGB_PARAMS.copy()
            if class_weight_dict and len(class_weight_dict) == 2:
                # Get the positive class (1) weight
                pos_class = label_encoder.transform([1])[0] if 1 in class_weight_dict else 1
                neg_class = 1 - pos_class
                xgb_params['scale_pos_weight'] = class_weight_dict[label_encoder.inverse_transform([pos_class])[0]] / \
                                               class_weight_dict[label_encoder.inverse_transform([neg_class])[0]]
            clf = XGBClassifier(**xgb_params)
            clf.fit(X_train, y_train_encoded)
        
        # Store the label encoder in the classifier for later use
        clf.label_encoder_ = label_encoder

    elif model_type == "lr":
        # Logistic Regression
        clf = LogisticRegression(
            **LOGISTIC_REGRESSION_PARAMS,
            class_weight=class_weight_dict
        )
        clf.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return clf


def aggregate_predictions_by_domain(domains, y_true, y_pred, targets_df):
    """
    Aggregate predictions by domain following the rules:
    1. If any prediction for a domain is 'partner', the domain is 'partner'
    2. If any prediction is 'partner-seo' and none are 'partner', the domain is 'partner-seo'
    3. Otherwise, use the majority prediction
    
    Use the true category from the targets dataset
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


def rank_models_by_domain_confusion_matrices():
    """
    Rank model configurations based on how well they predict 'partner' and 'partner-seo' classes.
    Focus on minimizing false negatives for these important classes.
    Save results as a CSV file.
    """
    logging.info("Ranking models based on domain confusion matrices...")
    
    # Get all domain prediction CSV files
    prediction_files = [f for f in os.listdir('results/output_predictions') if f.endswith('_domain.csv')]
    
    model_metrics = []
    
    for file in prediction_files:
        # Extract model prefix from filename
        prefix = file.replace('prediction_', '').replace('_domain.csv', '')
        
        # Load predictions
        df = pd.read_csv(f'results/output_predictions/{file}')
        
        # Calculate metrics focused on partner and partner-seo classes
        metrics = {}
        
        # Get true and predicted labels
        y_true = df['true_category']
        y_pred = df['predicted_category']
        
        # Create a confusion matrix for manual calculation
        # This is more robust than using recall_score for multilabel cases
        labels = sorted(set(list(y_true.unique()) + list(y_pred.unique())))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        
        # Calculate recall for 'partner' class: TP / (TP + FN)
        try:
            if 'partner' in cm_df.index:
                # For partner class: true positives / all actual partners
                partner_tp = cm_df.loc['partner', 'partner']
                partner_total = cm_df.loc['partner'].sum()
                partner_recall = partner_tp / partner_total if partner_total > 0 else 0
                metrics['partner_recall'] = partner_recall
                
                # Log misclassifications for debugging
                logging.info(f"{prefix} - Partner class: {partner_tp} correctly classified out of {partner_total} total")
                if partner_total > 0 and partner_tp < partner_total:
                    partner_errors = cm_df.loc['partner'].drop('partner')
                    for idx, val in partner_errors.items():
                        if val > 0:
                            logging.info(f"  {val} 'partner' samples misclassified as '{idx}'")
            else:
                metrics['partner_recall'] = 0
                logging.info(f"{prefix} - No 'partner' instances in test set")
                
            # Calculate recall for 'partner-seo' class
            if 'partner-seo' in cm_df.index:
                # For partner-seo class: true positives / all actual partner-seo
                seo_tp = cm_df.loc['partner-seo', 'partner-seo']
                seo_total = cm_df.loc['partner-seo'].sum()
                seo_recall = seo_tp / seo_total if seo_total > 0 else 0
                metrics['partner_seo_recall'] = seo_recall
                
                # Log misclassifications for debugging
                logging.info(f"{prefix} - Partner-SEO class: {seo_tp} correctly classified out of {seo_total} total")
                if seo_total > 0 and seo_tp < seo_total:
                    seo_errors = cm_df.loc['partner-seo'].drop('partner-seo')
                    for idx, val in seo_errors.items():
                        if val > 0:
                            logging.info(f"  {val} 'partner-seo' samples misclassified as '{idx}'")
            else:
                metrics['partner_seo_recall'] = 0
                logging.info(f"{prefix} - No 'partner-seo' instances in test set")
                
        except Exception as e:
            logging.warning(f"Error calculating metrics for {prefix}: {e}")
            metrics['partner_recall'] = 0
            metrics['partner_seo_recall'] = 0
            
        # Weighted combined score (prioritizing recall of important classes)
        # Higher weight for partner class
        weighted_score = (metrics.get('partner_recall', 0) * 0.7) + (metrics.get('partner_seo_recall', 0) * 0.3)
        metrics['weighted_score'] = weighted_score
        
        # Overall accuracy for reference
        metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)
        
        # Add to results
        model_metrics.append({
            'model': prefix,
            'partner_recall': metrics.get('partner_recall', 0),
            'partner_seo_recall': metrics.get('partner_seo_recall', 0),
            'weighted_score': metrics.get('weighted_score', 0),
            'overall_accuracy': metrics.get('overall_accuracy', 0)
        })
    
    # Create DataFrame and sort by weighted score (descending)
    metrics_df = pd.DataFrame(model_metrics)
    metrics_df = metrics_df.sort_values('weighted_score', ascending=False).reset_index(drop=True)
    
    # Save to CSV
    metrics_df.to_csv('results/model_rankings.csv', index=False)
    
    # Log results
    logging.info("Model rankings based on minimizing false negatives for partner classes:")
    logging.info("\n" + metrics_df.to_string())
    
    return metrics_df


def save_domain_confusion_matrix(domain_df, title, prefix):
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
    plt.savefig(f'results/Domains/confusion_matrix_{prefix}_test.png')
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
    plt.savefig(f'results/Domains/confusion_matrix_{prefix}_test_normalized.png')
    plt.close()
    
    # Save domain aggregation results to CSV with new naming convention
    domain_df.to_csv(f'results/output_predictions/prediction_{prefix}_domain.csv', index=False)


def evaluate_model(clf, X_test, y_test, domains, config, targets_df):
    """Evaluate model and save results"""
    # Make predictions
    if config["model_type"] == "xgb" and hasattr(clf, 'label_encoder_'):
        # For XGBoost, decode the predictions back to original labels
        y_pred_encoded = clf.predict(X_test)
        y_pred = clf.label_encoder_.inverse_transform(y_pred_encoded)
    else:
        y_pred = clf.predict(X_test)
    
    # Generate prefix based on the configuration
    prefix = f"{config['model_type']}_{'umap_' if config['use_umap'] else ''}{'smote_' if config['use_smote'] else ''}{'weights' if config['use_class_weights'] else 'base'}"
    
    # Save URL-level results
    url_results_df = pd.DataFrame({
        'domain': domains,
        'true_category': y_test,
        'predicted_category': y_pred
    })
    url_results_df.to_csv(f'results/output_predictions/prediction_{prefix}_url.csv', index=False)
    
    # Save URL-level confusion matrix
    save_confusion_matrix(
        y_test, y_pred, 
        f"Confusion Matrix ({prefix} - Test Set)",
        f"confusion_matrix_{prefix}_test"
    )
    
    # Print classification report for URL-level predictions
    print(f"\nClassification Report ({prefix} - Test Set, URL Level):")
    print(classification_report(y_test, y_pred))
    
    # Aggregate and evaluate domain-level predictions
    domain_df = aggregate_predictions_by_domain(domains, y_test, y_pred, targets_df)
    
    # Save domain-level predictions and confusion matrix
    save_domain_confusion_matrix(
        domain_df,
        f"Confusion Matrix ({prefix} - Test Set)",
        prefix
    )
    
    # Print classification report for domain-level predictions
    print(f"\nClassification Report ({prefix} - Test Set, Domain Level):")
    print(classification_report(domain_df['true_category'], domain_df['predicted_category']))
    
    return y_pred, prefix


def train_and_evaluate_model(X_train, X_test, y_train, y_test, domains_test, config, targets_df):
    """Train and evaluate model based on configuration"""
    # Train the model
    clf = train_model(X_train.copy(), y_train.copy(), config)
    
    # Evaluate the model
    y_pred, prefix = evaluate_model(clf, X_test, y_test, domains_test, config, targets_df)
    
    return clf, y_pred, prefix


def save_models(classifiers_dict):
    """Save all trained models"""
    for name, clf in classifiers_dict.items():
        joblib.dump(clf, f'models/classifier_model_{name}.joblib')


def save_training_artifacts(X_train, X_test, y_train, y_test, embedding_model):
    """Save training artifacts for future use"""
    # Save train/test split indices
    split_data = {
        'X_train': X_train.index.tolist(),
        'X_test': X_test.index.tolist(),
        'y_train': y_train.index.tolist(),
        'y_test': y_test.index.tolist()
    }
    with open('models/split_indices.json', 'w') as f:
        json.dump(split_data, f)
    
    # Save labels
    np.save('models/y_train.npy', y_train)
    np.save('models/y_test.npy', y_test)
    
    # Save embedding model
    embedding_model.save('models/embedding_model')


def log_results(classifiers_dict, train_data):
    """Log model parameters and dataset statistics"""
    with open('results/experiment_log.txt', 'w') as f:
        f.write("=== Dataset Statistics ===\n")
        f.write(f"Total samples: {len(train_data)}\n")
        f.write(f"Category distribution:\n{train_data['category'].value_counts().to_string()}\n\n")
        
        f.write("=== Model Parameters ===\n")
        f.write(f"Random Forest Parameters: {RANDOM_FOREST_PARAMS}\n")
        f.write(f"XGBoost Parameters: {XGB_PARAMS}\n")
        f.write(f"Logistic Regression Parameters: {LOGISTIC_REGRESSION_PARAMS}\n\n")
        
        f.write("=== Trained Models ===\n")
        for name, clf in classifiers_dict.items():
            f.write(f"\n{name.upper()} Model Type: {type(clf).__name__}\n")


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('results/training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def validate_data(texts: pd.DataFrame, targets: pd.DataFrame) -> None:
    """Validate input data format and content"""
    required_columns = {
        'texts': ['domain', 'description'],
        'targets': ['domain', 'category']
    }
    
    for df, cols in [(texts, required_columns['texts']), 
                     (targets, required_columns['targets'])]:
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")


def reduce_dimensions_with_umap(X_train_embeddings, X_test_embeddings, n_components=100):
    """Reduce dimensionality of embeddings using UMAP"""
    logging.info(f"Reducing dimensionality with UMAP to {n_components} components...")
    
    # Initialize and fit UMAP on training data
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    X_train_reduced = reducer.fit_transform(X_train_embeddings)
    
    # Transform test data using the fitted UMAP
    X_test_reduced = reducer.transform(X_test_embeddings)
    
    logging.info(f"Reduced embedding dimensions from {X_train_embeddings.shape[1]} to {n_components}")
    
    return reducer, X_train_reduced, X_test_reduced


def main():
    """Main execution function"""
    try:
        # Setup logging
        setup_logging()
        logging.info("Starting model training process")
        
        # Setup
        plt.rcParams['figure.figsize'] = (12, 8)
        setup_directories()
        
        # Load and prepare data
        logging.info("Loading data...")
        targets, features, texts = load_data("data")
        validate_data(texts, targets)
        
        train_data = prepare_data(texts, targets)
        logging.info(f"Prepared data shape: {train_data.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(train_data)
        
        # Get domains for test set
        domains_test = train_data.loc[X_test.index, 'domain']
        
        # Create embeddings
        logging.info("Creating embeddings...")
        embedding_model, X_train_embeddings, X_test_embeddings = create_embeddings(X_train, X_test)
        
        # Save original embeddings
        np.save('embeddings/train_embeddings_original.npy', X_train_embeddings)
        np.save('embeddings/test_embeddings_original.npy', X_test_embeddings)
        
        # Reduce dimensionality with UMAP
        umap_reducer, X_train_reduced, X_test_reduced = reduce_dimensions_with_umap(
            X_train_embeddings, X_test_embeddings, n_components=N_COMPONENTS_UMAP
        )
        
        # Save UMAP embeddings
        np.save('embeddings/train_embeddings_umap.npy', X_train_reduced)
        np.save('embeddings/test_embeddings_umap.npy', X_test_reduced)

        # Save the UMAP reducer
        joblib.dump(umap_reducer, 'models/umap_reducer.joblib')
        
        # Dictionary to store all classifiers
        classifiers = {}
        
        # Train and evaluate all models
        for config in MODEL_CONFIGS:
            model_type = config["model_type"]
            use_umap = config["use_umap"]
            use_smote = config["use_smote"]
            use_class_weights = config["use_class_weights"]
            
            config_desc = (f"{model_type.upper()} model "
                         f"{'with' if use_umap else 'without'} UMAP, "
                         f"{'with' if use_smote else 'without'} SMOTE, "
                         f"{'with' if use_class_weights else 'without'} class weights")
            
            logging.info(f"\n=== Training {config_desc} ===")
            
            # Select appropriate embeddings
            X_train_emb = X_train_reduced if use_umap else X_train_embeddings
            X_test_emb = X_test_reduced if use_umap else X_test_embeddings
            
            # Train and evaluate model
            clf, y_pred, prefix = train_and_evaluate_model(
                X_train_emb,
                X_test_emb,
                y_train,
                y_test,
                domains_test,
                config=config,
                targets_df=targets
            )
            classifiers[prefix] = clf
        
        # Rank models based on domain confusion matrices
        rank_models_by_domain_confusion_matrices()
        
        # Save all artifacts
        logging.info("Saving models and artifacts...")
        save_models(classifiers)
        save_training_artifacts(X_train, X_test, y_train, y_test, embedding_model)
        log_results(classifiers, train_data)
        
        logging.info("Training process completed successfully")

    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
