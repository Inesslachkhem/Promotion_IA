"""
Flask API Integration for Advanced AI Promotion Model
===================================================

This module integrates the advanced AI promotion model with Flask API
for seamless integration with .NET backend and Angular frontend.

Features:
- RESTful API endpoints
- Model training and retraining
- Real-time predictions
- Batch processing for categories
- Model performance monitoring
- Error handling and validation

Author: Smart Promotion System
Date: June 2025
"""

from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy import text
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import traceback
import pickle
import os
from ai_promotion_model import AdvancedPromotionAI, RetailKPICalculator
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Angular frontend

# SQL Server Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = (
    'mssql+pyodbc://DESKTOP-S22JEMV\\SQLEXPRESS/SmartPromoDb_v2024'
    '?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'ai-promotion-secret-key'

db = SQLAlchemy(app)

# Global AI model instance
ai_model = None
model_last_trained = None
kpi_calculator = RetailKPICalculator()

# Model persistence
MODEL_SAVE_PATH = 'models/ai_promotion_model.pkl'

def load_or_train_model():
    """
    Load existing model or train a new one
    """
    global ai_model, model_last_trained
    
    try:
        # Try to load existing model
        if os.path.exists(MODEL_SAVE_PATH):
            logger.info("Loading existing AI model...")
            with open(MODEL_SAVE_PATH, 'rb') as f:
                model_data = pickle.load(f)
                ai_model = model_data['model']
                model_last_trained = model_data['timestamp']
            logger.info(f"Model loaded successfully. Last trained: {model_last_trained}")
        else:
            logger.info("No existing model found. Training new model...")
            train_new_model()
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Training new model...")
        train_new_model()

def train_new_model():
    """
    Train a new AI model
    """
    global ai_model, model_last_trained
    
    try:
        logger.info("Initializing new AI model...")
        ai_model = AdvancedPromotionAI(db.session)
        
        logger.info("Training AI model...")
        metrics = ai_model.train_models(lookback_days=180)
        
        model_last_trained = datetime.now()
        
        # Save model
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        with open(MODEL_SAVE_PATH, 'wb') as f:
            pickle.dump({
                'model': ai_model,
                'timestamp': model_last_trained,
                'metrics': metrics
            }, f)
        
        logger.info("AI model trained and saved successfully")
        return metrics
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def validate_request_data(data, required_fields):
    """
    Validate request data
    """
    if not data:
        return False, "No data provided"
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    return True, None

# API Routes

@app.route('/')
def home():
    """
    API home page
    """
    global model_last_trained
    
    return jsonify({
        "message": "Advanced AI Promotion Model API",
        "version": "2.0",
        "model_status": "ready" if ai_model else "not_loaded",
        "last_trained": model_last_trained.isoformat() if model_last_trained else None,
        "endpoints": {
            "model_info": "GET /model-info",
            "train_model": "POST /train-model",
            "predict_product": "POST /predict-promotion",
            "predict_category": "POST /predict-category",
            "analyze_product": "GET /analyze-product/<article_id>",
            "kpi_calculation": "POST /calculate-kpis",
            "model_insights": "GET /model-insights"
        },
        "features": [
            "Real ML algorithms (XGBoost, Random Forest)",
            "Retail KPIs (Rotation, Sell-through rate, Stock coverage)",
            "Advanced feature engineering",
            "Model interpretability with SHAP",
            "Batch prediction capabilities"
        ]
    })

@app.route('/model-info')
def model_info():
    """
    Get model information and status
    """
    global ai_model, model_last_trained
    
    if not ai_model:
        return jsonify({
            "status": "error",
            "message": "Model not loaded"
        }), 500
    
    try:
        insights = ai_model.get_model_insights()
        
        return jsonify({
            "status": "success",
            "model_info": {
                "last_trained": model_last_trained.isoformat() if model_last_trained else None,
                "feature_count": insights.get("feature_count", 0),
                "models_trained": insights.get("models_trained", {}),
                "performance_metrics": insights.get("model_metrics", {}),
                "explainer_available": insights.get("explainer_available", False)
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/train-model', methods=['POST'])
def train_model():
    """
    Train or retrain the AI model
    """
    try:
        data = request.get_json() or {}
        lookback_days = data.get('lookback_days', 180)
        
        logger.info(f"Starting model training with {lookback_days} days lookback...")
        
        metrics = train_new_model()
        
        return jsonify({
            "status": "success",
            "message": "Model trained successfully",
            "training_time": model_last_trained.isoformat(),
            "metrics": metrics,
            "lookback_days": lookback_days
        })
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/predict-promotion', methods=['POST'])
def predict_promotion():
    """
    Predict promotion strategy for a specific product
    """
    if not ai_model:
        return jsonify({
            "status": "error",
            "message": "Model not loaded. Please train the model first."
        }), 500
    
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'article_id' not in data:
            return jsonify({
                "status": "error",
                "message": "article_id is required"
            }), 400
        
        article_id = data['article_id']
        
        # Make prediction
        prediction = ai_model.predict_promotion_strategy(article_id=article_id)
        
        if 'error' in prediction:
            return jsonify({
                "status": "error",
                "message": prediction['error']
            }), 404
        
        # Format response for frontend compatibility
        response = {
            "status": "success",
            "prediction": prediction,
            "recommendation": {
                "action": "promote" if prediction['needs_promotion'] else "no_action",
                "confidence": prediction['promotion_probability'],
                "discount_rate": prediction.get('recommended_discount', 0),
                "expected_impact": prediction.get('expected_sales_increase', 0)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error predicting promotion: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/predict-category', methods=['POST'])
def predict_category():
    """
    Predict promotion strategies for all products in a category
    """
    if not ai_model:
        return jsonify({
            "status": "error",
            "message": "Model not loaded. Please train the model first."
        }), 500
    
    try:
        data = request.get_json()
        
        # Validate input
        valid, error = validate_request_data(data, ['category_id'])
        if not valid:
            return jsonify({
                "status": "error",
                "message": error
            }), 400
        
        category_id = data['category_id']
        
        # Make batch predictions
        predictions = ai_model.batch_predict_category(category_id)
        
        # Analyze results
        total_products = len(predictions)
        products_needing_promotion = sum(1 for p in predictions if p.get('needs_promotion', False))
        average_discount = np.mean([p.get('recommended_discount', 0) for p in predictions if p.get('needs_promotion', False)]) if products_needing_promotion > 0 else 0
        
        return jsonify({
            "status": "success",
            "category_id": category_id,
            "summary": {
                "total_products": total_products,
                "products_needing_promotion": products_needing_promotion,
                "promotion_rate": products_needing_promotion / total_products if total_products > 0 else 0,
                "average_recommended_discount": average_discount
            },
            "predictions": predictions
        })
    
    except Exception as e:
        logger.error(f"Error predicting category: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/analyze-product/<int:article_id>')
def analyze_product(article_id):
    """
    Get detailed analysis of a specific product
    """
    try:
        # Get product basic info
        query = text("""
            SELECT 
                a.Id, a.CodeArticle, a.Libelle, a.Prix_Vente_TND, a.Prix_Achat_TND,
                a.IdCategorie, c.Nom as category_name,
                COALESCE(s.QuantitePhysique, 0) as current_stock
            FROM Articles a
            LEFT JOIN Categories c ON a.IdCategorie = c.IdCategorie
            LEFT JOIN Stocks s ON a.Id = s.ArticleId
            WHERE a.Id = :article_id
        """)
        
        result = db.session.execute(query, {"article_id": article_id})
        product_row = result.fetchone()
        
        if not product_row:
            return jsonify({
                "status": "error",
                "message": f"Product {article_id} not found"
            }), 404
        
        # Get sales data for KPI calculation
        sales_query = text("""
            SELECT 
                COUNT(*) as sales_count,
                COALESCE(SUM(v.QuantiteFacturee), 0) as total_quantity_sold,
                COALESCE(SUM(v.QuantiteFacturee * v.Prix_Vente_TND), 0) as total_revenue,
                MAX(v.Date) as last_sale_date,
                -- Recent sales (last 30 days)
                COALESCE(SUM(CASE WHEN v.Date >= DATEADD(day, -30, GETDATE()) 
                                 THEN v.QuantiteFacturee ELSE 0 END), 0) as recent_sales,
                -- Previous sales (30-60 days ago)
                COALESCE(SUM(CASE WHEN v.Date >= DATEADD(day, -60, GETDATE()) 
                                 AND v.Date < DATEADD(day, -30, GETDATE())
                                 THEN v.QuantiteFacturee ELSE 0 END), 0) as previous_sales
            FROM Ventes v
            INNER JOIN Stocks s ON v.StockId = s.Id
            WHERE s.ArticleId = :article_id
            AND v.Date >= DATEADD(day, -180, GETDATE())
        """)
        
        sales_result = db.session.execute(sales_query, {"article_id": article_id})
        sales_row = sales_result.fetchone()
        
        # Calculate KPIs
        current_price = float(product_row[3]) if product_row[3] else 0
        cost_price = float(product_row[4]) if product_row[4] else current_price * 0.7
        current_stock = int(product_row[7])
        total_sales = int(sales_row[1]) if sales_row[1] else 0
        recent_sales = int(sales_row[4]) if sales_row[4] else 0
        previous_sales = int(sales_row[5]) if sales_row[5] else 0
        
        # Estimate quantity injected (simplified)
        estimated_injected = max(current_stock + total_sales, 1)
        
        kpis = {
            "rotation": kpi_calculator.calculate_rotation(total_sales, estimated_injected),
            "sell_through_rate": kpi_calculator.calculate_sell_through_rate(total_sales, estimated_injected),
            "stock_coverage": kpi_calculator.calculate_stock_coverage(current_stock, total_sales / 180),
            "sales_trend": kpi_calculator.calculate_sales_trend(recent_sales, previous_sales),
            "profit_margin": kpi_calculator.calculate_profit_margin(current_price, cost_price)
        }
        
        # Get AI prediction if model is available
        ai_prediction = None
        if ai_model:
            try:
                ai_prediction = ai_model.predict_promotion_strategy(article_id=article_id)
            except Exception as e:
                logger.warning(f"Could not get AI prediction for {article_id}: {e}")
        
        return jsonify({
            "status": "success",
            "product_info": {
                "article_id": int(product_row[0]),
                "code_article": product_row[1],
                "product_name": product_row[2],
                "current_price": current_price,
                "cost_price": cost_price,
                "category_id": product_row[5],
                "category_name": product_row[6],
                "current_stock": current_stock
            },
            "sales_metrics": {
                "total_sales_180_days": total_sales,
                "recent_sales_30_days": recent_sales,
                "previous_sales_30_days": previous_sales,
                "last_sale_date": sales_row[3].isoformat() if sales_row[3] else None
            },
            "kpis": kpis,
            "ai_prediction": ai_prediction
        })
    
    except Exception as e:
        logger.error(f"Error analyzing product {article_id}: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/calculate-kpis', methods=['POST'])
def calculate_kpis():
    """
    Calculate KPIs for given product data
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['total_sales', 'quantity_injected', 'current_stock', 'avg_daily_sales']
        valid, error = validate_request_data(data, required_fields)
        if not valid:
            return jsonify({
                "status": "error",
                "message": error
            }), 400
        
        # Extract data
        total_sales = float(data['total_sales'])
        quantity_injected = float(data['quantity_injected'])
        current_stock = float(data['current_stock'])
        avg_daily_sales = float(data['avg_daily_sales'])
        recent_sales = float(data.get('recent_sales', 0))
        previous_sales = float(data.get('previous_sales', 0))
        selling_price = float(data.get('selling_price', 0))
        cost_price = float(data.get('cost_price', selling_price * 0.7))
        
        # Calculate all KPIs
        kpis = {
            "rotation": kpi_calculator.calculate_rotation(total_sales, quantity_injected),
            "sell_through_rate": kpi_calculator.calculate_sell_through_rate(total_sales, quantity_injected),
            "stock_coverage": kpi_calculator.calculate_stock_coverage(current_stock, avg_daily_sales),
            "sales_trend": kpi_calculator.calculate_sales_trend(recent_sales, previous_sales),
            "profit_margin": kpi_calculator.calculate_profit_margin(selling_price, cost_price)
        }
        
        # Add interpretations
        interpretations = {
            "rotation": "High" if kpis["rotation"] > 1.0 else "Medium" if kpis["rotation"] > 0.5 else "Low",
            "sell_through_rate": "Excellent" if kpis["sell_through_rate"] > 80 else "Good" if kpis["sell_through_rate"] > 60 else "Poor",
            "stock_coverage": "Overstocked" if kpis["stock_coverage"] > 90 else "Healthy" if kpis["stock_coverage"] > 30 else "Low Stock",
            "sales_trend": "Growing" if kpis["sales_trend"] > 0.1 else "Declining" if kpis["sales_trend"] < -0.1 else "Stable",
            "profit_margin": "High" if kpis["profit_margin"] > 0.4 else "Medium" if kpis["profit_margin"] > 0.2 else "Low"
        }
        
        return jsonify({
            "status": "success",
            "kpis": kpis,
            "interpretations": interpretations,
            "recommendations": generate_kpi_recommendations(kpis)
        })
    
    except Exception as e:
        logger.error(f"Error calculating KPIs: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def generate_kpi_recommendations(kpis):
    """
    Generate recommendations based on KPI values
    """
    recommendations = []
    
    if kpis["rotation"] < 0.5:
        recommendations.append("Low rotation detected. Consider promotion to increase sales velocity.")
    
    if kpis["stock_coverage"] > 90:
        recommendations.append("High stock coverage. Promotion recommended to reduce inventory.")
    
    if kpis["sales_trend"] < -0.2:
        recommendations.append("Declining sales trend. Urgent action needed.")
    
    if kpis["sell_through_rate"] < 50:
        recommendations.append("Low sell-through rate. Review pricing and promotion strategy.")
    
    if not recommendations:
        recommendations.append("Product performance is healthy. No immediate action required.")
    
    return recommendations

@app.route('/model-insights')
def model_insights():
    """
    Get detailed model insights and feature importance
    """
    if not ai_model:
        return jsonify({
            "status": "error",
            "message": "Model not loaded"
        }), 500
    
    try:
        insights = ai_model.get_model_insights()
        
        return jsonify({
            "status": "success",
            "insights": insights,
            "interpretation": {
                "feature_importance": "Features ranked by their impact on promotion decisions",
                "model_performance": "Accuracy metrics for each prediction model",
                "data_quality": f"Model trained on {insights.get('feature_count', 0)} features"
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting model insights: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Health check endpoint
@app.route('/health')
def health_check():
    """
    API health check
    """
    try:
        # Test database connection
        db.session.execute(text("SELECT 1"))
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "model": "loaded" if ai_model else "not_loaded"
        })
    
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "available_endpoints": [
            "/model-info", "/train-model", "/predict-promotion",
            "/predict-category", "/analyze-product/<id>", "/calculate-kpis"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error",
        "message": "Internal server error",
        "suggestion": "Check server logs for details"
    }), 500

# Initialize the model on startup
@app.before_first_request
def initialize_model():
    """
    Initialize the AI model when the app starts
    """
    logger.info("Initializing AI model...")
    load_or_train_model()

if __name__ == '__main__':
    print("🚀 Starting Advanced AI Promotion API")
    print("=" * 60)
    print("Features:")
    print("✅ Advanced ML algorithms for promotion optimization")
    print("✅ Real-time prediction capabilities")
    print("✅ Retail KPI calculations")
    print("✅ Model interpretability and insights")
    print("✅ RESTful API for easy integration")
    print("✅ CORS enabled for Angular frontend")
    print("=" * 60)
    
    # Initialize model
    with app.app_context():
        load_or_train_model()
    
    app.run(debug=True, host='0.0.0.0', port=5001)
