# AI vs Human Text Detector

A fine-tuned DistilBERT model that classifies whether a text is human-written or AI-generated. The project includes a Flask web interface and SHAP-based explainability for transparent predictions.

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers (DistilBERT)
* SHAP
* Flask
* Scikit-learn

## Features

* Fine-tuned DistilBERT classifier
* Long-text support through chunked inference
* SHAP explainability for word-level insights
* Flask-based web interface
* Confidence score visualization

## Dataset

AI vs Human Text Dataset (Kaggle)
100,000 balanced samples (50k Human, 50k AI)

## Performance

* Accuracy: 99.3%
* ROC-AUC: 0.9997
* F1 Score: 0.993

## Project Structure

* `train_distilbert.py` — Model training
* `predict.py` — Terminal inference
* `app.py` — Flask web application
* `config.py` — Configuration settings

## Limitations

Performance may decrease on highly formal or academic human-written text due to dataset bias.

## Note: While the model achieved very high evaluation metrics, the results are likely influenced by dataset-specific patterns and overfitting, limiting real-world generalization; this project was primarily undertaken as a learning experience in NLP model training, deployment, and explainable AI.

**Author:** [Lourembam Rivaldo]
**Institution:** Assam Don Bosco University]
**Year:** 2026
