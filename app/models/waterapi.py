# app/models/waterapi.py
import sys
import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import database functions with proper error handling
try:
    # Add the parent directory to Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from app.database.database_main import get_latest_water_data
    DATABASE_AVAILABLE = True
    logger.info("✅ Database module imported successfully")
except ImportError as e:
    logger.warning(f"❌ Database module not available: {e}")
    DATABASE_AVAILABLE = False
    # Create dummy function for fallback
    def get_latest_water_data(*args, **kwargs):
        return []

class WaterQualityPredictor:
    """Water quality prediction and analysis class"""
    
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load pre-trained model or train a new one"""
        try:
            # Try to load pre-trained model
            model_path = os.path.join(os.path.dirname(__file__), '..', 'rfr_actuator.joblib')
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info("✅ Pre-trained model loaded successfully")
            else:
                logger.warning("❌ Pre-trained model not found. Using default predictions.")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            self.model = None
    
    def predict_water_quality(self, ph, turbidity, temperature, tds):
        """Predict water quality based on parameters"""
        try:
            if self.model:
                # Use the trained model for prediction
                features = np.array([[ph, turbidity, temperature, tds]])
                prediction = self.model.predict(features)[0]
                probability = self.model.predict_proba(features)[0]
                
                return {
                    "potability": float(probability[1] if len(probability) > 1 else 0.8),
                    "quality": "safe" if prediction == 1 else "unsafe",
                    "confidence": float(max(probability)),
                    "parameters": {
                        "ph": ph,
                        "turbidity": turbidity,
                        "temperature": temperature,
                        "tds": tds
                    }
                }
            else:
                # Fallback prediction logic
                return self._fallback_prediction(ph, turbidity, temperature, tds)
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._fallback_prediction(ph, turbidity, temperature, tds)
    
    def _fallback_prediction(self, ph, turbidity, temperature, tds):
        """Fallback prediction when model is not available"""
        # Simple rule-based prediction
        score = 0
        
        # pH scoring (ideal: 6.5-8.5)
        if 6.5 <= ph <= 8.5:
            score += 0.3
        elif 6.0 <= ph <= 9.0:
            score += 0.2
        else:
            score += 0.1
        
        # Turbidity scoring (lower is better, ideal: < 5 NTU)
        if turbidity < 5:
            score += 0.3
        elif turbidity < 10:
            score += 0.2
        else:
            score += 0.1
        
        # Temperature scoring (ideal: 10-30°C)
        if 10 <= temperature <= 30:
            score += 0.2
        else:
            score += 0.1
        
        # TDS scoring (ideal: < 500 ppm)
        if tds < 500:
            score += 0.2
        elif tds < 1000:
            score += 0.1
        else:
            score += 0.05
        
        # Normalize score to 0-1 range
        potability = min(max(score, 0), 1)
        
        return {
            "potability": potability,
            "quality": "safe" if potability > 0.6 else "unsafe",
            "confidence": 0.7,
            "parameters": {
                "ph": ph,
                "turbidity": turbidity,
                "temperature": temperature,
                "tds": tds
            },
            "warning": "Using fallback prediction - model not available"
        }
    
    def get_water_analysis(self, device_id=None, limit=10):
        """Get comprehensive water analysis"""
        try:
            # Get recent data
            if DATABASE_AVAILABLE:
                recent_data = get_latest_water_data(device_id, limit)
            else:
                recent_data = []
            
            analysis = {
                "recent_readings": recent_data,
                "statistics": self._calculate_statistics(recent_data),
                "trends": self._analyze_trends(recent_data),
                "recommendations": []
            }
            
            # Add predictions for each reading
            for reading in recent_data:
                reading['prediction'] = self.predict_water_quality(
                    reading.get('ph_value', 7.0),
                    reading.get('turbidity', 5.0),
                    reading.get('temperature', 25.0),
                    reading.get('tds_value', 120.0)
                )
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_recommendations(analysis['statistics'])
            
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "recent_readings": [],
                "statistics": {},
                "trends": {},
                "recommendations": []
            }
    
    def _calculate_statistics(self, data):
        """Calculate statistics from water data"""
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        stats = {}
        
        for param in ['ph_value', 'turbidity', 'temperature', 'tds_value']:
            if param in df.columns:
                stats[param] = {
                    'mean': float(df[param].mean()),
                    'median': float(df[param].median()),
                    'min': float(df[param].min()),
                    'max': float(df[param].max()),
                    'std': float(df[param].std())
                }
        
        return stats
    
    def _analyze_trends(self, data):
        """Analyze trends in water data"""
        if len(data) < 2:
            return {}
        
        df = pd.DataFrame(data)
        trends = {}
        
        for param in ['ph_value', 'turbidity', 'temperature', 'tds_value']:
            if param in df.columns and 'time' in df.columns:
                try:
                    df_sorted = df.sort_values('time')
                    values = df_sorted[param].values
                    if len(values) > 1:
                        trend = "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable"
                        change = abs(values[-1] - values[0])
                        trends[param] = {
                            'trend': trend,
                            'change': float(change),
                            'rate': float(change / len(values))
                        }
                except:
                    continue
        
        return trends
    
    def _generate_recommendations(self, statistics):
        """Generate recommendations based on statistics"""
        recommendations = []
        
        # pH recommendations
        ph_stats = statistics.get('ph_value', {})
        if ph_stats:
            ph_mean = ph_stats.get('mean', 7.0)
            if ph_mean < 6.5:
                recommendations.append("pH is too low. Consider adding alkaline substances.")
            elif ph_mean > 8.5:
                recommendations.append("pH is too high. Consider adding acidic substances.")
        
        # Turbidity recommendations
        turbidity_stats = statistics.get('turbidity', {})
        if turbidity_stats:
            turbidity_mean = turbidity_stats.get('mean', 5.0)
            if turbidity_mean > 5:
                recommendations.append("High turbidity detected. Consider filtration.")
        
        # Temperature recommendations
        temp_stats = statistics.get('temperature', {})
        if temp_stats:
            temp_mean = temp_stats.get('mean', 25.0)
            if temp_mean > 30:
                recommendations.append("Water temperature is high. Monitor for bacterial growth.")
        
        # TDS recommendations
        tds_stats = statistics.get('tds_value', {})
        if tds_stats:
            tds_mean = tds_stats.get('mean', 120.0)
            if tds_mean > 500:
                recommendations.append("High TDS detected. Consider reverse osmosis treatment.")
        
        if not recommendations:
            recommendations.append("Water quality parameters are within acceptable ranges.")
        
        return recommendations

# Create global instance
water_predictor = WaterQualityPredictor()

# API functions
def predict_water_quality(ph, turbidity, temperature, tds):
    """Predict water quality from parameters"""
    return water_predictor.predict_water_quality(ph, turbidity, temperature, tds)

def get_water_analysis(device_id=None, limit=10):
    """Get comprehensive water analysis"""
    return water_predictor.get_water_analysis(device_id, limit)

def train_new_model(data_path=None):
    """Train a new model (optional)"""
    # This would be implemented if you want to retrain the model
    logger.info("Model training functionality would be implemented here")
    return {"status": "training not implemented"}

# Test the module
if __name__ == "__main__":
    print("Testing Water Quality Predictor...")
    
    # Test prediction
    result = predict_water_quality(7.0, 5.0, 25.0, 120.0)
    print(f"Prediction result: {result}")
    
    # Test analysis
    analysis = get_water_analysis()
    print(f"Analysis result keys: {list(analysis.keys())}")
    
    print("✅ Water API module loaded successfully!")
    