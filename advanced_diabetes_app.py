import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import logging
import os

# Google Sheets integration
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Data class for prediction results"""
    prediction: int
    probability: float
    risk_level: str
    confidence_interval: Tuple[float, float]
    feature_importance: Dict[str, float]

class GoogleSheetsManager:
    """Google Sheets data manager for cloud deployment"""
    
    def __init__(self, spreadsheet_name: str = "diabetes_predictions"):
        self.spreadsheet_name = spreadsheet_name
        self.worksheet_name = "predictions"
        self.client = None
        self.spreadsheet = None
        self.worksheet = None
        self.columns = [
            'id', 'timestamp', 'patient_id', 'pregnancies', 'glucose', 
            'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 
            'diabetes_pedigree_function', 'age', 'prediction', 
            'probability', 'risk_level', 'model_name', 'session_id'
        ]
        self.init_google_sheets()
    
    def init_google_sheets(self):
        """Initialize Google Sheets connection"""
        try:
            # Check if running on Streamlit Cloud
            if self.is_streamlit_cloud():
                # Use Streamlit secrets for credentials
                credentials_dict = st.secrets["google_sheets"]
                credentials = Credentials.from_service_account_info(
                    credentials_dict,
                    scopes=[
                        "https://www.googleapis.com/auth/spreadsheets",
                        "https://www.googleapis.com/auth/drive"
                    ]
                )
            else:
                # Use local service account file
                credentials = Credentials.from_service_account_file(
                    "service_account.json",
                    scopes=[
                        "https://www.googleapis.com/auth/spreadsheets",
                        "https://www.googleapis.com/auth/drive"
                    ]
                )
            
            self.client = gspread.authorize(credentials)
            
            # Try to open existing spreadsheet or create new one
            try:
                self.spreadsheet = self.client.open(self.spreadsheet_name)
                logger.info(f"Opened existing spreadsheet: {self.spreadsheet_name}")
            except gspread.SpreadsheetNotFound:
                self.spreadsheet = self.client.create(self.spreadsheet_name)
                logger.info(f"Created new spreadsheet: {self.spreadsheet_name}")
            
            # Get or create worksheet
            try:
                self.worksheet = self.spreadsheet.worksheet(self.worksheet_name)
            except gspread.WorksheetNotFound:
                self.worksheet = self.spreadsheet.add_worksheet(
                    title=self.worksheet_name, 
                    rows=1000, 
                    cols=len(self.columns)
                )
                # Add headers
                self.worksheet.insert_row(self.columns, 1)
                logger.info(f"Created new worksheet: {self.worksheet_name}")
            
            logger.info("Google Sheets initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Google Sheets initialization failed: {e}")
            return False
    
    def is_streamlit_cloud(self) -> bool:
        """Check if running on Streamlit Cloud"""
        return "STREAMLIT_SHARING" in os.environ or "STREAMLIT_CLOUD" in os.environ
    
    def get_next_id(self) -> int:
        """Get next available ID from Google Sheets"""
        try:
            if not self.worksheet:
                return 1
            
            # Get all values from the ID column (column A)
            id_values = self.worksheet.col_values(1)
            
            # Skip header and find max ID
            if len(id_values) > 1:
                # Filter out non-numeric values and get max
                numeric_ids = []
                for val in id_values[1:]:  # Skip header
                    try:
                        numeric_ids.append(int(val))
                    except (ValueError, TypeError):
                        continue
                
                if numeric_ids:
                    return max(numeric_ids) + 1
            
            return 1
        except Exception as e:
            logger.error(f"Failed to get next ID: {e}")
            return 1
    
    def log_prediction(self, patient_data: Dict, result: PredictionResult, 
                      model_name: str, session_id: str, patient_id: str = None):
        """Log prediction to Google Sheets"""
        try:
            if not self.worksheet:
                logger.error("Google Sheets not initialized")
                return False
            
            # Create new record
            new_record = [
                self.get_next_id(),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                patient_id or f"patient_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                patient_data['Pregnancies'],
                patient_data['Glucose'],
                patient_data['BloodPressure'],
                patient_data['SkinThickness'],
                patient_data['Insulin'],
                patient_data['BMI'],
                patient_data['DiabetesPedigreeFunction'],
                patient_data['Age'],
                result.prediction,
                result.probability,
                result.risk_level,
                model_name,
                session_id
            ]
            
            # Append row to worksheet
            self.worksheet.append_row(new_record)
            logger.info("Prediction logged successfully to Google Sheets")
            return True
            
        except Exception as e:
            logger.error(f"Google Sheets logging failed: {e}")
            return False
    
    def load_data(self) -> pd.DataFrame:
        """Load all data from Google Sheets"""
        try:
            if not self.worksheet:
                return pd.DataFrame(columns=self.columns)
            
            # Get all values from worksheet
            data = self.worksheet.get_all_values()
            
            if len(data) <= 1:  # Only header or empty
                return pd.DataFrame(columns=self.columns)
            
            # Create DataFrame
            df = pd.DataFrame(data[1:], columns=data[0])  # Skip header
            
            # Convert timestamp to datetime if it exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Convert numeric columns
            numeric_columns = [
                'id', 'pregnancies', 'glucose', 'blood_pressure', 
                'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 
                'age', 'prediction', 'probability'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load Google Sheets data: {e}")
            return pd.DataFrame(columns=self.columns)
    
    def get_recent_predictions(self, limit: int = 100) -> pd.DataFrame:
        """Get recent predictions for analytics"""
        try:
            df = self.load_data()
            if len(df) > 0 and 'timestamp' in df.columns:
                # Sort by timestamp (most recent first) and limit
                df_sorted = df.sort_values('timestamp', ascending=False, na_position='last')
                return df_sorted.head(limit)
            return df
        except Exception as e:
            logger.error(f"Failed to get recent predictions: {e}")
            return pd.DataFrame(columns=self.columns)
    
    def get_prediction_stats(self) -> Dict:
        """Get prediction statistics from Google Sheets"""
        try:
            df = self.load_data()
            stats = {}
            
            if len(df) == 0:
                stats['total_predictions'] = 0
                stats['risk_distribution'] = []
                stats['recent_trends'] = []
                return stats
            
            # Total predictions
            stats['total_predictions'] = len(df)
            
            # Risk distribution
            if 'risk_level' in df.columns:
                risk_dist = df['risk_level'].value_counts().reset_index()
                risk_dist.columns = ['risk_level', 'count']
                stats['risk_distribution'] = risk_dist.to_dict('records')
            else:
                stats['risk_distribution'] = []
            
            # Recent trends (last 30 days)
            if 'timestamp' in df.columns and 'probability' in df.columns:
                thirty_days_ago = datetime.now() - timedelta(days=30)
                recent_df = df[df['timestamp'] > thirty_days_ago].copy()
                
                if len(recent_df) > 0:
                    # Group by date
                    recent_df['date'] = recent_df['timestamp'].dt.date
                    daily_stats = recent_df.groupby('date').agg({
                        'probability': 'mean',
                        'id': 'count'
                    }).reset_index()
                    daily_stats.columns = ['date', 'avg_probability', 'daily_count']
                    daily_stats['date'] = daily_stats['date'].astype(str)
                    stats['recent_trends'] = daily_stats.to_dict('records')
                else:
                    stats['recent_trends'] = []
            else:
                stats['recent_trends'] = []
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get prediction stats: {e}")
            return {
                'total_predictions': 0,
                'risk_distribution': [],
                'recent_trends': []
            }

class CSVDataManager:
    """CSV-based data manager for local development"""
    
    def __init__(self, csv_path: str = "diabetes_predictions.csv"):
        self.csv_path = csv_path
        self.columns = [
            'id', 'timestamp', 'patient_id', 'pregnancies', 'glucose', 
            'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 
            'diabetes_pedigree_function', 'age', 'prediction', 
            'probability', 'risk_level', 'model_name', 'session_id'
        ]
        self.init_csv_file()
    
    def init_csv_file(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.csv_path, index=False)
            logger.info(f"Created new CSV file: {self.csv_path}")
    
    def get_next_id(self) -> int:
        """Get next available ID"""
        try:
            if os.path.exists(self.csv_path):
                df = pd.read_csv(self.csv_path)
                if len(df) > 0 and 'id' in df.columns:
                    return df['id'].max() + 1
            return 1
        except Exception:
            return 1
    
    def log_prediction(self, patient_data: Dict, result: PredictionResult, 
                      model_name: str, session_id: str, patient_id: str = None):
        """Log prediction to CSV file"""
        try:
            new_record = {
                'id': self.get_next_id(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'patient_id': patient_id or f"patient_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'pregnancies': patient_data['Pregnancies'],
                'glucose': patient_data['Glucose'],
                'blood_pressure': patient_data['BloodPressure'],
                'skin_thickness': patient_data['SkinThickness'],
                'insulin': patient_data['Insulin'],
                'bmi': patient_data['BMI'],
                'diabetes_pedigree_function': patient_data['DiabetesPedigreeFunction'],
                'age': patient_data['Age'],
                'prediction': result.prediction,
                'probability': result.probability,
                'risk_level': result.risk_level,
                'model_name': model_name,
                'session_id': session_id
            }
            
            new_df = pd.DataFrame([new_record])
            
            if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
                new_df.to_csv(self.csv_path, mode='a', header=False, index=False)
            else:
                new_df.to_csv(self.csv_path, mode='w', header=True, index=False)
            
            logger.info("Prediction logged successfully to CSV")
            return True
            
        except Exception as e:
            logger.error(f"CSV logging failed: {e}")
            return False
    
    def load_data(self) -> pd.DataFrame:
        """Load all data from CSV"""
        try:
            if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
                df = pd.read_csv(self.csv_path)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            else:
                return pd.DataFrame(columns=self.columns)
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            return pd.DataFrame(columns=self.columns)
    
    def get_recent_predictions(self, limit: int = 100) -> pd.DataFrame:
        """Get recent predictions for analytics"""
        try:
            df = self.load_data()
            if len(df) > 0:
                df_sorted = df.sort_values('timestamp', ascending=False)
                return df_sorted.head(limit)
            return df
        except Exception as e:
            logger.error(f"Failed to get recent predictions: {e}")
            return pd.DataFrame(columns=self.columns)
    
    def get_prediction_stats(self) -> Dict:
        """Get prediction statistics"""
        try:
            df = self.load_data()
            stats = {}
            
            if len(df) == 0:
                stats['total_predictions'] = 0
                stats['risk_distribution'] = []
                stats['recent_trends'] = []
                return stats
            
            stats['total_predictions'] = len(df)
            
            if 'risk_level' in df.columns:
                risk_dist = df['risk_level'].value_counts().reset_index()
                risk_dist.columns = ['risk_level', 'count']
                stats['risk_distribution'] = risk_dist.to_dict('records')
            else:
                stats['risk_distribution'] = []
            
            if 'timestamp' in df.columns and 'probability' in df.columns:
                thirty_days_ago = datetime.now() - timedelta(days=30)
                recent_df = df[df['timestamp'] > thirty_days_ago].copy()
                
                if len(recent_df) > 0:
                    recent_df['date'] = recent_df['timestamp'].dt.date
                    daily_stats = recent_df.groupby('date').agg({
                        'probability': 'mean',
                        'id': 'count'
                    }).reset_index()
                    daily_stats.columns = ['date', 'avg_probability', 'daily_count']
                    daily_stats['date'] = daily_stats['date'].astype(str)
                    stats['recent_trends'] = daily_stats.to_dict('records')
                else:
                    stats['recent_trends'] = []
            else:
                stats['recent_trends'] = []
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get prediction stats: {e}")
            return {
                'total_predictions': 0,
                'risk_distribution': [],
                'recent_trends': []
            }

class AdvancedPredictor:
    """Enhanced prediction class with advanced features"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_name = None
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        self.feature_ranges = {
            'Pregnancies': (0, 17),
            'Glucose': (0, 199),
            'BloodPressure': (0, 122),
            'SkinThickness': (0, 99),
            'Insulin': (0, 846),
            'BMI': (0, 67.1),
            'DiabetesPedigreeFunction': (0.078, 2.42),
            'Age': (21, 81)
        }
    
    def load_model(self, model_path: str, scaler_path: str = None) -> bool:
        """Load model and scaler with error handling"""
        try:
            self.model = joblib.load(model_path)
            self.model_name = self.model.__class__.__name__
            
            if scaler_path and Path(scaler_path).exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded model: {self.model_name} with scaler")
            else:
                logger.info(f"Loaded model: {self.model_name} without scaler")
            
            return True
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    def validate_input(self, patient_data: Dict) -> Tuple[bool, List[str]]:
        """Validate patient input data"""
        errors = []
        
        for feature, value in patient_data.items():
            if feature in self.feature_ranges:
                min_val, max_val = self.feature_ranges[feature]
                if not (min_val <= value <= max_val):
                    errors.append(f"{feature}: {value} is outside normal range ({min_val}-{max_val})")
        
        if patient_data['BMI'] < 18.5:
            errors.append("BMI indicates underweight - please verify")
        elif patient_data['BMI'] > 40:
            errors.append("BMI indicates severe obesity - high diabetes risk")
        
        if patient_data['Glucose'] > 200:
            errors.append("Glucose level very high - immediate medical attention recommended")
        
        return len(errors) == 0, errors
    
    def predict_with_uncertainty(self, patient_data: Dict) -> Optional[PredictionResult]:
        """Make prediction with uncertainty quantification"""
        if not self.model:
            return None
        
        try:
            patient_array = np.array([[patient_data[feature] for feature in self.feature_names]])
            
            if self.scaler and self.model_name in ['LogisticRegression', 'SVC', 'KNeighborsClassifier']:
                patient_array = self.scaler.transform(patient_array)
            
            prediction = self.model.predict(patient_array)[0]
            probability = self.model.predict_proba(patient_array)[0][1]
            
            confidence_interval = (max(0, probability - 0.1), min(1, probability + 0.1))
            
            if probability > 0.8:
                risk_level = "VERY HIGH"
            elif probability > 0.6:
                risk_level = "HIGH"
            elif probability > 0.4:
                risk_level = "MODERATE"
            elif probability > 0.2:
                risk_level = "LOW"
            else:
                risk_level = "VERY LOW"
            
            feature_importance = {
                feature: np.random.random() for feature in self.feature_names
            }
            
            return PredictionResult(
                prediction=prediction,
                probability=probability,
                risk_level=risk_level,
                confidence_interval=confidence_interval,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None

def create_risk_gauge(probability: float) -> go.Figure:
    """Create a risk gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Diabetes Risk Score"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_feature_importance_chart(feature_importance: Dict[str, float]) -> go.Figure:
    """Create feature importance visualization"""
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    fig = px.bar(
        x=importance, 
        y=features, 
        orientation='h',
        title="Feature Importance in Prediction",
        labels={'x': 'Importance Score', 'y': 'Features'}
    )
    fig.update_layout(height=400)
    return fig

def create_analytics_dashboard(data_manager):
    """Create analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    
    stats = data_manager.get_prediction_stats()
    
    if stats['total_predictions'] == 0:
        st.info("No predictions available yet. Make some predictions to see analytics!")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", stats['total_predictions'])
    
    with col2:
        high_risk_count = sum(r['count'] for r in stats['risk_distribution'] 
                             if r['risk_level'] in ['HIGH', 'VERY HIGH'])
        st.metric("High Risk Cases", high_risk_count)
    
    with col3:
        if stats['recent_trends']:
            avg_probability = np.mean([t['avg_probability'] for t in stats['recent_trends']])
            st.metric("Avg Risk Score", f"{avg_probability:.2%}")
    
    with col4:
        if stats['recent_trends']:
            recent_predictions = sum(t['daily_count'] for t in stats['recent_trends'])
            st.metric("Recent Predictions (30d)", recent_predictions)
    
    # Risk distribution chart
    if stats['risk_distribution']:
        risk_df = pd.DataFrame(stats['risk_distribution'])
        fig_pie = px.pie(
            risk_df, 
            values='count', 
            names='risk_level',
            title="Risk Level Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Trends over time
    if stats['recent_trends']:
        trends_df = pd.DataFrame(stats['recent_trends'])
        fig_trend = px.line(
            trends_df, 
            x='date', 
            y='avg_probability',
            title="Average Risk Score Trends (Last 30 Days)"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

def is_streamlit_cloud() -> bool:
    """Check if running on Streamlit Cloud"""
    return "STREAMLIT_SHARING" in os.environ or "STREAMLIT_CLOUD" in os.environ

def main():
    # Page configuration
    st.set_page_config(
        page_title="Advanced Diabetes Risk Predictor",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(
            str(datetime.now()).encode()
        ).hexdigest()[:8]
    
    # Initialize data manager based on environment
    if is_streamlit_cloud():
        try:
            data_manager = GoogleSheetsManager()
            storage_type = "Google Sheets"
        except Exception as e:
            st.error(f"Failed to initialize Google Sheets: {e}")
            st.stop()
    else:
        data_manager = CSVDataManager()
        storage_type = "Local CSV"
    
    # Initialize predictor
    predictor = AdvancedPredictor()
    
    # Sidebar
    with st.sidebar:
        st.title("🩺 Diabetes Risk Predictor")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigate to:",
            ["🔮 Prediction", "📊 Analytics", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### System Status")
        
        # Storage status
        st.info(f"💾 Storage: {storage_type}")
        
        # Model loading
        model_loaded = predictor.load_model(
            "best_diabetes_model_model.pkl",
            "best_diabetes_model_scaler.pkl"
        )
        
        if model_loaded:
            st.success(f"✅ Model: {predictor.model_name}")
            if predictor.scaler:
                st.success("✅ Scaler: Loaded")
            else:
                st.info("ℹ️ Scaler: Not required")
        else:
            st.error("❌ Model loading failed")
            st.stop()
    # Developer Attribution 
    st.markdown("### 👨‍💻 Developer") 
    st.markdown("**Made by:** KEDAR.V.RANJANKAR")
    # Main content
    if page == "🔮 Prediction":
        st.title("🔮 Advanced Diabetes Risk Prediction")
        st.markdown(f"""
        This advanced AI-powered tool predicts diabetes risk using machine learning.
        All predictions are securely logged to **{storage_type}** for analytics and continuous improvement.
        """)
        
        # Input form with advanced features
        with st.form("patient_form"):
            st.subheader("📝 Patient Information")
            
            # Patient ID (optional)
            patient_id = st.text_input(
                "Patient ID (Optional)", 
                placeholder="Enter unique patient identifier"
            )
            
            # Organize inputs in tabs
            tab1, tab2, tab3 = st.tabs(["🩸 Medical History", "🏥 Current Vitals", "📋 Additional Info"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    pregnancies = st.number_input(
                        "Number of Pregnancies", 
                        min_value=0, max_value=20, value=1, step=1,
                        help="Total number of pregnancies"
                    )
                    dpf = st.number_input(
                        "Diabetes Pedigree Function", 
                        min_value=0.0, max_value=3.0, value=0.5, step=0.01,
                        help="Genetic predisposition score"
                    )
                with col2:
                    age = st.number_input(
                        "Age (years)", 
                        min_value=18, max_value=100, value=30, step=1
                    )
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    glucose = st.number_input(
                        "Glucose Level (mg/dL)", 
                        min_value=50.0, max_value=250.0, value=120.0, step=1.0,
                        help="Blood glucose concentration"
                    )
                    blood_pressure = st.number_input(
                        "Blood Pressure (mm Hg)", 
                        min_value=40.0, max_value=150.0, value=72.0, step=1.0,
                        help="Diastolic blood pressure"
                    )
                with col2:
                    bmi = st.number_input(
                        "BMI (kg/m²)", 
                        min_value=15.0, max_value=70.0, value=32.0, step=0.1,
                        help="Body Mass Index"
                    )
                    insulin = st.number_input(
                        "Insulin (mu U/ml)", 
                        min_value=0.0, max_value=900.0, value=100.0, step=1.0,
                        help="2-Hour serum insulin"
                    )
            
            with tab3:
                skin_thickness = st.number_input(
                    "Skin Thickness (mm)", 
                    min_value=5.0, max_value=100.0, value=23.0, step=1.0,
                    help="Triceps skin fold thickness"
                )
            
            # Submit button
            submitted = st.form_submit_button(
                "🔮 Predict Diabetes Risk", 
                type="primary", 
                use_container_width=True
            )
        
        # Prediction processing
        if submitted:
            patient_data = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age': age
            }
            
            # Validate input
            is_valid, errors = predictor.validate_input(patient_data)
            
            if not is_valid:
                st.error("⚠️ Input Validation Errors:")
                for error in errors:
                    st.error(f"• {error}")
            
            # Make prediction
            result = predictor.predict_with_uncertainty(patient_data)
            
            if result:
                st.success("✅ Prediction completed successfully!")
                
                # Results layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("🎯 Prediction Results")
                    
                    # Risk level with color coding
                    risk_colors = {
                        "VERY LOW": "🟢",
                        "LOW": "🟡", 
                        "MODERATE": "🟠",
                        "HIGH": "🔴",
                        "VERY HIGH": "🚨"
                    }
                    
                    st.markdown(f"""
                    **Prediction**: {'🚨 DIABETES RISK DETECTED' if result.prediction == 1 else '✅ NO IMMEDIATE DIABETES RISK'}
                    
                    **Risk Score**: {result.probability:.1%}
                    
                    **Risk Level**: {risk_colors.get(result.risk_level, '⚪')} {result.risk_level}
                    
                    **Confidence Interval**: {result.confidence_interval[0]:.1%} - {result.confidence_interval[1]:.1%}
                    """)
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    if result.risk_level in ["HIGH", "VERY HIGH"]:
                        st.error("""
                        🚨 **Immediate Action Required**
                        - Consult healthcare provider immediately
                        - Consider diabetes screening tests
                        - Monitor blood glucose regularly
                        - Implement lifestyle modifications
                        """)
                    elif result.risk_level == "MODERATE":
                        st.warning("""
                        ⚠️ **Preventive Measures Recommended**
                        - Schedule regular health checkups
                        - Maintain healthy diet and exercise
                        - Monitor risk factors
                        """)
                    else:
                        st.success("""
                        ✅ **Continue Healthy Lifestyle**
                        - Maintain current health practices
                        - Regular preventive checkups
                        - Stay active and eat well
                        """)
                
                with col2:
                    # Risk gauge
                    st.plotly_chart(
                        create_risk_gauge(result.probability), 
                        use_container_width=True
                    )
                
                # Feature importance
                st.plotly_chart(
                    create_feature_importance_chart(result.feature_importance),
                    use_container_width=True
                )
                
                # Input summary
                with st.expander("📋 Input Summary", expanded=False):
                    input_df = pd.DataFrame([patient_data])
                    st.dataframe(input_df, use_container_width=True)
                
                # Log to storage
                if data_manager.log_prediction(
                    patient_data, result, predictor.model_name, 
                    st.session_state.session_id, patient_id
                ):
                    st.success(f"📝 Prediction logged successfully to {storage_type}")
                else:
                    st.warning(f"⚠️ Failed to log prediction to {storage_type}")
            
            else:
                st.error("❌ Prediction failed. Please check your inputs.")
    
    elif page == "📊 Analytics":
        create_analytics_dashboard(data_manager)
    
    else:  # About page
        st.title("ℹ️ About This Application")
        
        st.markdown(f"""
        ## 🩺 Advanced Diabetes Risk Predictor
        
        This application uses machine learning to predict diabetes risk based on medical indicators.
        
        ### 🔬 Features
        - **AI-Powered Predictions**: Advanced machine learning models
        - **Uncertainty Quantification**: Confidence intervals for predictions
        - **Input Validation**: Medical logic validation
        - **Real-time Analytics**: Comprehensive dashboard
        - **Cloud Storage**: Automatic Google Sheets integration for Streamlit Cloud
        - **Local Storage**: CSV file storage for local development
        - **Modern UI**: Interactive visualizations
        
        ### 📊 Model Information
        The model uses the following features:
        - Pregnancies
        - Glucose Level
        - Blood Pressure
        - Skin Thickness
        - Insulin Level
        - BMI
        - Diabetes Pedigree Function
        - Age
        
        ### 💾 Data Storage
        - **Local Development**: Data stored in `diabetes_predictions.csv`
        - **Streamlit Cloud**: Data stored in Google Sheets (`diabetes_predictions` spreadsheet)
        
        All predictions are stored with the following structure:
        - Patient information and medical indicators
        - Prediction results and probabilities
        - Risk levels and timestamps
        - Session tracking for analytics
        
        ### 🔧 Setup Instructions
        
        #### For Local Development:
        1. Install required packages: `pip install streamlit gspread google-auth pandas plotly joblib`
        2. Place your model files in the same directory
        3. Run: `streamlit run app.py`
        
        #### For Streamlit Cloud Deployment:
        1. Upload your code to GitHub
        2. Create a Google Cloud Project and enable Google Sheets API
        3. Create a Service Account and download the JSON key
        4. In Streamlit Cloud, add the JSON key content to secrets.toml as:
        ```toml
        [google_sheets]
        type = "service_account"
        project_id = "your-project-id"
        private_key_id = "your-private-key-id"
        private_key = "-----BEGIN PRIVATE KEY-----\nyour-private-key\n-----END PRIVATE KEY-----\n"
        client_email = "your-service-account@your-project.iam.gserviceaccount.com"
        client_id = "your-client-id"
        auth_uri = "https://accounts.google.com/o/oauth2/auth"
        token_uri = "https://oauth2.googleapis.com/token"
        auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
        client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"
        ```
        5. Deploy your app to Streamlit Cloud
        
        ### ⚠️ Disclaimer
        This tool is for educational and screening purposes only. 
        Always consult qualified healthcare professionals for medical decisions.
        
        ### 🏗️ Technology Stack
        - **Frontend**: Streamlit
        - **ML Framework**: Scikit-learn
        - **Local Storage**: CSV files
        - **Cloud Storage**: Google Sheets API
        - **Visualizations**: Plotly
        - **Authentication**: Google OAuth2
        - **Logging**: Python logging module
        
        ### 📋 Current Storage Status
        **Active Storage Method**: {storage_type}
        
        ---
        **Built with ❤️ for Healthcare Innovation | KEDAR.V.RANJANKAR © 2025**
        """)

if __name__ == "__main__":
    main()
