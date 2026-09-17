================================================================================
WINE QUALITY PREDICTION & INTERACTIVE AI ASSISTANT: PROJECT README
================================================================================

1. PROJECT DESCRIPTION
--------------------------------------------------------------------------------
- What the application does: 
  This application provides an end-to-end machine learning and AI-powered interface 
  for predicting wine quality based on physicochemical properties (such as acidity, 
  sugar, chlorides, and alcohol content). It combines a robust regression pipeline 
  with an intelligent LLM interface that parses natural language descriptions into 
  structured feature inputs for real-time quality scoring.

- Who it is for: 
  It is designed for enologists, wine production facilities, quality control 
  engineers, and educators who need reliable quality estimations and interactive 
  exploratory analysis of red and white wine profiles.

- What problem it solves: 
  Traditional wine tasting and laboratory quality grading can be subjective, 
  expensive, and time-consuming. This system automates the evaluation process, 
  allowing users to instantly predict wine scores from lab metrics or query the 
  system using natural conversational language to evaluate hypothetical wine formulations.


2. SETUP INSTRUCTIONS
--------------------------------------------------------------------------------
- Prerequisites:
  * Python 3.10+
  * Git & DVC (Data Version Control)

- Step 1: Clone the Repository
  git clone <https://github.com/c9johnson/capstone-wine-quality>
  cd capstone-wine-quality

- Step 2: Install Dependencies
  pip install --upgrade pip
  pip install -r requirements.txt

- Step 3: Configure API Keys
  Create a `.env` file in the root directory or configure environment variables 
  for your LLM provider tokens:
  
  HF_TOKEN=your_huggingface_token_here

- Step 4: Get the Data & Artifacts
  * If working locally with DVC tracking:
    dvc pull
  * The dataset is automatically constructed and verified during pipeline 
    execution and testing environments.


3. USAGE INSTRUCTIONS
--------------------------------------------------------------------------------
- Training the Model:
  Run the complete training pipeline to preprocess data, train models, track 
  metrics via MLflow, and log the best champion model:
  python src/train.py

- Running Inference:
  Execute predictions on sample or custom wine features using the prediction module:
  python src/predict.py

- Running the Test Suite:
  Validate model performance and inference output shapes using pytest:
  pytest

- Exploring with Open WebUI / Local AI:
  Launch your custom containerized interface or web app to interact with the 
  model through natural language prompts.


4. ARCHITECTURE OVERVIEW
--------------------------------------------------------------------------------
- Data Preprocessing Pipeline (`src/preprocess.py`):
  Ingests raw UCI red and white wine CSVs, standardizes column structures, merges 
  them into a unified dataset incorporating a `wine_type` categorical feature, and 
  splits the data into training and testing partitions.

- Machine Learning Model (`src/train.py` & MLflow):
  Trains regression models on the physicochemical feature space. All experiments, 
  hyperparameters, and evaluation metrics are tracked locally via MLflow, logging 
  the top-performing artifact as the champion model (`models/best_model.pkl`).

- LLM Integration & Prediction Layer (`src/predict.py`):
  Bridges user intent with the machine learning backend. When a user provides a 
  natural language description of a wine, the LLM extracts the relevant chemical 
  features into a structured dictionary. The inference pipeline applies 
  standardized preprocessing rules, handles default value imputations, and feeds 
  the feature vector into the champion model to return a scalar quality prediction (1-10).


5. RESULTS SUMMARY
--------------------------------------------------------------------------------
- Best Model Performance:
  The optimized champion model achieved strong predictive accuracy on unseen 
  test data, maintaining a low Root Mean Squared Error (RMSE) well within the 
  project baseline threshold (< 3.0 on the standard 1-10 scale), with high 
  reliability across both red and white wine subsets.

- Interesting Findings:
  * Alcohol content and volatile acidity emerged as the two most dominant predictors 
    influencing overall perceived wine quality.
  * Combining red and white wine datasets with an explicit `wine_type` indicator 
    allowed the model to capture shared chemical relationships while adjusting 
    baseline score expectations between varietals.


6. REFLECTION
--------------------------------------------------------------------------------
- What We Learned:
  Successfully integrating traditional tabular machine learning pipelines with 
  unstructured natural language processing layers requires strict schema validation 
  and fault-tolerant feature imputation to ensure smooth end-to-end execution.

- What Was Challenging:
  Bridging the gap between a stateful local development environment (with local 
  SQLite tracking stores and manual file dependencies) and a stateless cloud CI 
  runner (GitHub Actions) presented classic reproducibility hurdles, particularly 
  around data availability and artifact tracking.

- What We Would Improve With More Time:
  * Implement an automated cloud storage backend (such as S3 or Hugging Face Hub) 
    for DVC remote storage to streamline artifact sharing across team members 
    and CI environments.
  * Expand the LLM prompt parser to handle a wider range of conversational edge 
    cases and unit conversions.