# Diabetes Prediction Project

## Overview
This project develops a machine learning system to predict diabetes risk using the Pima Indian Diabetes Dataset (or synthetic data). It includes data preprocessing, model training, evaluation, hyperparameter tuning, and deployment via interactive applications. The project is divided into three parts, each focusing on specific stages of the machine learning pipeline. Two applications—a Streamlit web app and a Jupyter-native `ipywidgets` app—allow users to input patient details, select a model, and predict diabetes risk, with all predictions logged to `log.csv`.

## Features
- **Data Preprocessing**: Handles missing values, outliers, and feature scaling.
- **Model Training**: Trains seven models: Logistic Regression, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Naive Bayes, Decision Tree, and Gradient Boosting.
- **Model Evaluation**: Uses accuracy, precision, recall, F1-score, and ROC-AUC for performance assessment.
- **Hyperparameter Tuning**: Optimizes model performance using grid search or random search.
- **Feature Importance**: Analyzes key predictors of diabetes risk.
- **Model Saving**: Saves all trained models as separate `.pkl` files (e.g., `random_forest_model.pkl`).
- **Interactive Apps**:
  - **Streamlit App**: Web-based interface for model selection, patient data input, and prediction with logging.
  - **Jupyter App**: `ipywidgets`-based interface for use within Jupyter Notebook.
- **Prediction Logging**: Saves inputs, model used, prediction, probability, risk level, and timestamp to `log.csv`.
- **Google Sheets Integration**: Support for cloud-based data storage and sharing via Google Sheets API.
- **Enhanced Visualizations**: Interactive plots and charts using Plotly for better data insights.

## Requirements
- Python 3.8+
- Core Libraries:
  ```bash
  pip install streamlit>=1.28.0 numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=1.0.0 joblib>=1.1.0 plotly>=5.0.0 ipywidgets matplotlib seaborn
  ```
- Google Sheets Integration (Optional):
  ```bash
  pip install gspread>=5.7.0 google-auth>=2.15.0 gspread-dataframe>=3.3.0
  ```
- Additional Dependencies:
  ```bash
  pip install pathlib2>=2.3.0
  ```
- Jupyter Notebook (for running `.py` files as notebooks or using the `ipywidgets` app)
- Dataset: `diabetes.csv` (included or generated synthetically in Part 1)

## Setup
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd diabetes_prediction
   ```

2. **Install Dependencies**:
   
   **Option A: Using requirements.txt** (Recommended):
   ```bash
   pip install -r requirements.txt
   ```
   
   **Option B: Manual Installation**:
   ```bash
   # Core dependencies
   pip install streamlit>=1.28.0 numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=1.0.0 joblib>=1.1.0 plotly>=5.0.0 ipywidgets matplotlib seaborn
   
   # Google Sheets integration (optional)
   pip install gspread>=5.7.0 google-auth>=2.15.0 gspread-dataframe>=3.3.0
   
   # Additional utilities
   pip install pathlib2>=2.3.0
   ```

3. **Create requirements.txt** (if not provided):
   Create a `requirements.txt` file with the following content:
   ```
   streamlit>=1.28.0
   numpy>=1.21.0
   pandas>=1.3.0
   scikit-learn>=1.0.0
   joblib>=1.1.0
   plotly>=5.0.0
   ipywidgets>=7.6.0
   matplotlib>=3.3.0
   seaborn>=0.11.0
   gspread>=5.7.0
   google-auth>=2.15.0
   gspread-dataframe>=3.3.0
   pathlib2>=2.3.0
   ```

4. **Prepare Dataset**:
   - Ensure `diabetes.csv` is in the project directory. If using synthetic data, run `diabetes_part1.py` to generate it.

5. **Enable Jupyter Widgets** (for `ipywidgets` app):
   ```bash
   jupyter nbextension enable --py widgetsnbextension
   ```

6. **Google Sheets Setup** (Optional):
   - Create a Google Cloud Project and enable the Google Sheets API
   - Download the service account JSON credentials
   - Place the credentials file in your project directory
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable:
     ```bash
     export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
     ```

## Usage
### Running the Project
1. **Part 1: Data Preparation (`diabetes_part1.py`)**:
   - Run all cells in a Jupyter Notebook to load, preprocess, and analyze the dataset.
   - Outputs: Cleaned dataset, initial visualizations, and prepared features.
   ```python
   %run diabetes_part1.py
   ```

2. **Part 2: Model Training and Evaluation (`diabetes_part2_complete 1.py`)**:
   - Run all cells to train and evaluate seven models, select the best model, and save it as `best_diabetes_model_model.pkl`.
   ```python
   %run diabetes_part2_complete 1.py
   ```

3. **Part 3: Advanced Analysis and Deployment (`diabetes_part3.py`)**:
   - Run all cells for hyperparameter tuning, cross-validation, feature importance, and interactive prediction examples.
   - Generates `model_results` with trained models.
   ```python
   %run diabetes_part3.py
   ```

4. **Save All Models**:
   - After running Part 3, execute the following in a new cell to save all models:
     ```python
     import joblib
     for model_name, model_info in model_results.items():
         safe_model_name = model_name.lower().replace(' ', '_')
         joblib.dump(model_info['model'], f"{safe_model_name}_model.pkl")
         print(f"Saved {safe_model_name}_model.pkl")
     ```
   - Creates files like `random_forest_model.pkl`, `logistic_regression_model.pkl`, etc.
   - Note: Models requiring scaling (Logistic Regression, SVM, KNN) need `diabetes_scaler.pkl` (see Troubleshooting).

### Running the Applications
#### Streamlit App
1. **Save the Script**:
   - Copy `advanced_diabetes_app.py` to the project directory.

2. **Run the App**:
   ```python
   import subprocess
   subprocess.run(["streamlit", "run", "advanced_diabetes_app.py"])
   ```
   - Opens at `http://localhost:8501`.

3. **Interact**:
   - Select a model from the dropdown (e.g., Random Forest).
   - Enter patient details (Pregnancies, Glucose, etc.).
   - View enhanced visualizations powered by Plotly.
   - Click "Predict Diabetes Risk" to view prediction, probability, risk level (HIGH/MODERATE/LOW), and input features.
   - Predictions are logged to `log.csv` and optionally to Google Sheets.

4. **Stop the App**:
   - Interrupt the cell (Ctrl+C) or close the terminal.

#### Jupyter-Native App
1. **Run the Script**:
   - Copy `diabetes_prediction_jupyter_all_models.py` into a Jupyter Notebook cell and execute.

2. **Interact**:
   - Select a model from the dropdown.
   - Adjust sliders for patient details.
   - Click "Predict Diabetes Risk" to see results in the notebook.
   - Predictions are logged to `log.csv`.

3. **View Log**:
   ```python
   import pandas as pd
   print(pd.read_csv("log.csv"))
   ```

### Log File
- **Location**: `log.csv` in the project directory.
- **Columns**:
  - `Timestamp`: Date and time (e.g., `2025-05-30 00:19:00`)
  - `Model`: Selected model (e.g., `Random Forest`)
  - `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`: Input features
  - `Prediction`: `DIABETES RISK` or `NO DIABETES`
  - `Probability`: Diabetes probability (0.0–1.0)
  - `Risk_Level`: `HIGH` (>70%), `MODERATE` (30–70%), `LOW` (<30%)
- **Example**:
  ```
  Timestamp,Model,Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Prediction,Probability,Risk_Level
  2025-05-30 00:19:00,Random Forest,5,180.0,90.0,35.0,200.0,35.0,0.8,55,DIABETES RISK,0.8923,HIGH
  ```

## Troubleshooting
- **Model Results Not Defined**:
  - Ensure `train_and_evaluate_models` ran in Part 3. Rerun:
    ```python
    from diabetes_part3 import train_and_evaluate_models
    model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    ```

- **Missing Scaler**:
  - Models like Logistic Regression, SVM, and KNN require `diabetes_scaler.pkl`. Recreate it:
    ```python
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import joblib

    diabetes_data = pd.read_csv("diabetes.csv")
    zero_replacement_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for feature in zero_replacement_features:
        diabetes_data[feature] = diabetes_data[feature].replace(0, np.nan)
        diabetes_data[feature] = diabetes_data[feature].fillna(diabetes_data[feature].median())
    X = diabetes_data.drop('Outcome', axis=1)
    y = diabetes_data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, "diabetes_scaler.pkl")
    ```
  - Non-scaling models (Random Forest, Gradient Boosting, Decision Tree, Naive Bayes) work without it.

- **Streamlit Issues**:
  - Verify installation: `streamlit --version`.
  - Check port 8501: `lsof -i :8501`.
  - Ensure Streamlit version compatibility: `pip install --upgrade streamlit>=1.28.0`.

- **ipywidgets Not Displaying**:
  - Enable widgets: `jupyter nbextension enable --py widgetsnbextension`.
  - For JupyterLab: `jupyter labextension install @jupyter-widgets/jupyterlab-manager`.

- **Plotly Visualization Issues**:
  - Ensure Plotly is properly installed: `pip install --upgrade plotly>=5.0.0`.
  - For Jupyter: `pip install "notebook>=5.3" "ipywidgets>=7.2"`.

- **Google Sheets Integration Issues**:
  - Verify Google credentials are properly set up.
  - Check API quotas and permissions.
  - Ensure gspread and dependencies are up to date.

- **Log File Errors**:
  - Check write permissions: `pd.DataFrame([{'test': 1}]).to_csv('log.csv')`.
  - Ensure no file locks: `lsof log.csv`.

- **Pathlib2 Import Issues**:
  - This package provides backports for older Python versions. If using Python 3.8+, the built-in `pathlib` should suffice.
  - For compatibility issues, ensure pathlib2>=2.3.0 is installed.

## Future Improvements
- Add more sophisticated visualizations using Plotly's advanced features.
- Implement real-time Google Sheets synchronization for collaborative predictions.
- Support batch predictions from CSV files with progress tracking.
- Add model performance comparison dashboards in the Streamlit app.
- Implement automated model retraining workflows.
- Add support for additional cloud storage backends (AWS S3, Azure Blob).
- Containerize the project using Docker for improved portability.
- Add comprehensive unit tests and CI/CD pipelines.

## Contributors
- KEDAR .V. RANJANKAR

## License
This project is licensed under the MIT License.

## Acknowledgments
- Pima Indian Diabetes Dataset
- Scikit-learn, Streamlit, Plotly, and Jupyter communities
- Google Sheets API for cloud integration capabilities
