# Domain Classification for Keboola

This project provides scripts to train a model for classifying domains based on their descriptions, and then make predictions using the trained model. The scripts are specifically designed to work within the Keboola data platform.

## Scripts Overview

There are two main scripts:

1. **train_best_model_keboola.py** - Trains a model on the complete dataset and saves the model files
2. **predict_keboola.py** - Loads a trained model and makes predictions on new domain data

## Input/Output Structure

### For Training (train_best_model_keboola.py)

#### Input Tables:
- `/data/in/tables/keyword-analysis-mapping.csv` - Contains domain mappings with the 'domain' and 'category' columns
- `/data/in/tables/domains.csv` - Contains domains with the 'domain' and 'description' columns

#### Output Tables:
- `/data/out/tables/url_level_predictions.csv` - URL-level predictions on the training data
- `/data/out/tables/domain_level_predictions.csv` - Domain-level aggregated predictions
- `/data/out/tables/key_metrics.csv` - Evaluation metrics (recall, accuracy, etc.)
- `/data/out/tables/domains_predictions.csv` - Predictions for all domains with descriptions
- `/data/out/tables/domains_predictions_summary.csv` - Summary of predictions by category

#### Output Files:
- `/data/out/files/best_classifier_model.joblib` - The trained classifier model
- `/data/out/files/embedding_model/` - The sentence transformer embedding model
- `/data/out/files/class_weights.json` - Class weights used in training
- `/data/out/files/model_config.json` - Model configuration details
- `/data/out/files/confusion_matrices/` - Generated confusion matrix images
- `/data/out/files/training_log.log` - Detailed training log

### For Prediction (predict_keboola.py)

#### Input Tables:
- `/data/in/tables/domains.csv` - Contains domains with the 'domain' and 'description' columns

#### Input Files:
- `/data/in/files/best_classifier_model.joblib` - The trained classifier model
- `/data/in/files/embedding_model/` - The sentence transformer embedding model

#### Output Tables:
- `/data/out/tables/url_level_predictions.csv` - URL-level predictions
- `/data/out/tables/domain_level_predictions.csv` - Domain-level aggregated predictions
- `/data/out/tables/domains_predictions_summary.csv` - Summary of predictions by category

#### Output Files:
- `/data/out/tables/prediction_log.txt` - Detailed prediction log

## Keboola Configuration

### Training Component

1. Create a Keboola component with Python 3.7+
2. Upload the `train_best_model_keboola.py` script
3. Configure input mappings:
   - Map your domain data to `/data/in/tables/domains.csv`
   - Map your domain category mappings to `/data/in/tables/keyword-analysis-mapping.csv`
4. Configure output mappings:
   - Map all files in `/data/out/files/` to your storage
   - Map all tables in `/data/out/tables/` to your storage

### Prediction Component

1. Create a Keboola component with Python 3.7+
2. Upload the `predict_keboola.py` script
3. Configure input mappings:
   - Map your new domain data to `/data/in/tables/domains.csv`
   - Map the previously saved model files to `/data/in/files/`
4. Configure output mappings:
   - Map all tables in `/data/out/tables/` to your storage

## Dependencies

The scripts require the following Python packages:
- pandas
- numpy
- scikit-learn
- sentence-transformers
- joblib
- matplotlib
- seaborn

These should be installed in the Keboola Python environment.

## Prediction Process

The domain classification follows these rules for domain-level aggregation:
1. If any URL for a domain is classified as 'partner', the domain is classified as 'partner'
2. If any URL is classified as 'partner-seo' and none are 'partner', the domain is classified as 'partner-seo'
3. Otherwise, the majority classification is used

## Troubleshooting

If you encounter issues:
1. Check the log files for detailed error information
2. Ensure all required input files are present with the correct format
3. Verify that the file paths in the input/output mappings match the paths expected by the scripts 