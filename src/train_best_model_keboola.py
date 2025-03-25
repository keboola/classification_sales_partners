import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from keboola.component import CommonInterface

ci = CommonInterface() 
input_tables = ci.get_input_tables_definitions()

# Keboola folders with updated file paths
INPUT_PATH = 'in/tables/'
OUTPUT_PATH = 'out/tables/'

# Input file names
TEXTS_FILE = 'data_domains_classification.csv'  # Contains URLs with descriptions
TARGETS_FILE = 'domains_train.csv'  # Contains domain categories

# Output file names
URL_PREDICTIONS_FILE = 'classification_predictions_url.csv'
DOMAIN_PREDICTIONS_FILE = 'classification_predictions_domain.csv'

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


def load_data():
    """Load the data files from Keboola input tables"""
    texts_file = os.path.join(INPUT_PATH, TEXTS_FILE)
    targets_file = os.path.join(INPUT_PATH, TARGETS_FILE)
    
    if not os.path.exists(texts_file):
        raise FileNotFoundError(f"Texts file not found: {texts_file}")
    if not os.path.exists(targets_file):
        raise FileNotFoundError(f"Targets file not found: {targets_file}")
    
    # Load datasets
    texts = pd.read_csv(texts_file)
    targets = pd.read_csv(targets_file)
    
    # Check if required columns exist
    required_texts_columns = ['domain', 'description']
    required_targets_columns = ['domain', 'category']
    
    missing_texts_columns = [col for col in required_texts_columns if col not in texts.columns]
    missing_targets_columns = [col for col in required_targets_columns if col not in targets.columns]
    
    if missing_texts_columns:
        raise ValueError(f"Missing required columns in texts data: {missing_texts_columns}")
    if missing_targets_columns:
        raise ValueError(f"Missing required columns in targets data: {missing_targets_columns}")
    
    return texts, targets


def prepare_data(texts, targets):
    """Prepare and merge data for training"""
    # Filter texts with non-null and non-empty descriptions
    texts = texts[texts['description'].notna() & (texts['description'] != '')]
    
    # Merge and prepare training data
    data = pd.merge(texts, targets, on='domain', how='right')
    data['category'] = data['category_y']
    data = data.drop(['category_x', 'category_y'], axis=1)

    # Remove rows with NaN or empty descriptions
    data = data.dropna(subset=['description'])
    data = data[data['description'] != '']

    return data


def create_embeddings(X):
    """Create embeddings using SentenceTransformer"""
    embedding_model = SentenceTransformer(MODEL_NAME_TRANSFORMER)
    X_embeddings = embedding_model.encode(X.tolist(), show_progress_bar=True)
    return embedding_model, X_embeddings


def train_model(X_embeddings, y):
    """Train logistic regression with class weights"""
    # Calculate class weights using scikit-learn
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=pd.Series(y).unique(),
        y=y
    )
    class_weight_dict = dict(zip(pd.Series(y).unique(), class_weights))
    
    # Train Logistic Regression model with class weights
    clf = LogisticRegression(
        **LOGISTIC_REGRESSION_PARAMS,
        class_weight=class_weight_dict
    )
    clf.fit(X_embeddings, y)
    
    return clf

def aggregate_predictions_by_domain(url_predictions_df):
    """
    Aggregate predictions by domain following the rules:
    1. If any prediction for a domain is 'partner', the domain is 'partner'
    2. If any prediction is 'partner-seo' and none are 'partner', the domain is 'partner-seo'
    3. Otherwise, use the majority prediction
    """
    # Group by domain to aggregate results
    domain_results = []
    
    for domain, group in url_predictions_df.groupby('domain'):
        # Get all predictions for this domain
        predictions = group['predicted_category'].tolist()
        
        # Count occurrences of each category using pandas
        prediction_counts = pd.Series(predictions).value_counts()
        total_urls = len(predictions)
        
        # Determine aggregated prediction based on rules
        if 'partner' in prediction_counts:
            agg_prediction = 'partner'
        elif 'partner-seo' in prediction_counts:
            agg_prediction = 'partner-seo'
        else:
            # Use most common prediction (mode)
            agg_prediction = pd.Series(predictions).mode().iloc[0]
        
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

def predict_on_all_domains(texts, targets, clf, embedding_model):
    """Make predictions on all domains and save to Keboola output"""
    # Filter texts with non-null and non-empty descriptions
    texts_filtered = texts[texts['description'].notna() & (texts['description'] != '')]

    # Create embeddings for descriptions
    data_embeddings = embedding_model.encode(texts_filtered['description'].tolist(), show_progress_bar=True)

    # Make predictions
    predictions = clf.predict(data_embeddings)
    
    # Add predictions to the dataframe
    texts_filtered['predicted_category'] = predictions
    
    # Save URL-level predictions
    url_output_df = texts_filtered[['url', 'domain', 'description', 'predicted_category']]
    url_pred_path = os.path.join(OUTPUT_PATH, URL_PREDICTIONS_FILE)
    url_output_df.to_csv(url_pred_path, index=False)
    
    # Create domain-level predictions with aggregation
    domain_df = aggregate_predictions_by_domain(texts_filtered)
    
    # Save domain-level predictions
    domain_pred_path = os.path.join(OUTPUT_PATH, DOMAIN_PREDICTIONS_FILE)
    domain_df.to_csv(domain_pred_path, index=False)


"""Main execution function for Keboola integration"""
try:
    # Load data
    texts, targets = load_data()
    
    # Prepare training data
    data = prepare_data(texts, targets)
    
    # Get features and labels from the training dataset
    X = data['description']
    y = data['category']
    
    # Create embeddings
    embedding_model, X_embeddings = create_embeddings(X)
    
    # Train the model on training dataset
    clf = train_model(X_embeddings, y)
    
    # Make predictions on all domains
    predict_on_all_domains(texts, targets, clf, embedding_model)

except Exception as e:
    raise Exception(f"Error during training or prediction: {str(e)}")
