## Sentiment Emotion Stress Analyzer 

This project analyzes text input and predicts sentiment, emotion, and stress level using Machine Learning. It demonstrates how NLP and ML models can be used together to understand human language and mental state patterns.

## Algorithm 
- TF-IDF Vectorization: Converts raw text into a numerical matrix based on word importance.

- Multinomial Naive Bayes: A probabilistic learning algorithm used for categorizing text into emotions.

- Label Mapping: Translates numeric dataset labels into human-readable stress categories.

## Features 
- Real-Time Analysis: Instant prediction from user command-line input.

- Multi-Label Detection: Predicts specific emotions (Joy, Sadness, Anger, Fear, etc.).

- Stress Level Assessment: Converts emotional intensity into a stress score.

- Hybrid Data Loading: Seamlessly combines .csv and .parquet data sources.

- Visualization: Uses Matplotlib and Seaborn to show model accuracy and data distribution.

## Technologies Used 
- Language: Python 3.x
- Data Handling: Pandas, NumPy, PyArrow
- Machine Learning: Scikit-learn
- Visualization: Matplotlib, Seaborn
- Storage: Pickle (for model serialization) 

## Workflow 
- Data Ingestion: Loads CSV and Parquet files and standardizes column names.
- Preprocessing: Normalizes text (lowercasing) and maps numeric labels to emotion words.
- Vectorization: Transforms text into TF-IDF features.
- Training: Splits data (80/20) and trains a Pipeline model.
- Evaluation: Generates an Accuracy Score and Classification Report.
- Inference: Takes user input, runs it through the saved model, and outputs the result.

## How to Run 
- Clone the project folder to your local machine.

- Install dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn pyarrow

- Run the application:
python main.py

- Interact: Enter any sentence (e.g., "I feel overwhelmed with work") to see the analysis.

## Dataset
- Emotion Dataset (Parquet file)
- Sentiment Dataset (CSV/Excel file)

## Output
The system provides three levels of output for every text input:

1. Emotion Classification
The model identifies the primary emotional state of the text.

2. Performance Metrics
After training, the system displays a detailed evaluation of how well it learned the datasets.

- Accuracy Score: Shows the overall percentage of correct predictions (currently 87.42%).

- Classification Report: Provides precision, recall, and f1-scores for each individual category (Joy, Sadness, Anger, etc.).

3. Stress Level Indicators
The system translates emotional intensity into a numerical stress value to help quantify mental state patterns.

- Scale: 0 to 10 (where 0 is low stress/high joy and 10 is critical stress/anxiety).

- Visualization: Graphical insights including bar charts and pie charts that display the distribution of emotions across the entire dataset.
