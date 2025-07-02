"""
Flask API for AI Promotion Recommendation System
==============================================

This Flask API exposes the AI promotion functionality from ai_demo.py
as REST endpoints that can be consumed by the Angular frontend.

Endpoints:
- POST /api/promotions/generate - Generate promotion recommendations
- GET /api/promotions/categories - Get available product categories
- GET /api/health - Health check endpoint

Run: python flask_api.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
from datetime import datetime, timedelta
import traceback
import pandas as pd
import numpy as np

# Import our AI promotion demo class
from ai_demo import AIPromotionDemo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Angular frontend

# Initialize AI promotion demo instance (global variable for reuse)
ai_demo = None


def initialize_ai_demo():
    """Initialize the AI demo instance"""
    global ai_demo
    try:
        ai_demo = AIPromotionDemo()
        logger.info("✅ AI Promotion Demo initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI Demo: {e}")
        return False


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "AI Promotion API",
            "version": "1.0.0",
        }
    )


@app.route("/api/promotions/categories", methods=["GET"])
def get_categories():
    """Get available product categories"""
    try:
        global ai_demo
        if ai_demo is None:
            if not initialize_ai_demo():
                return jsonify({"error": "AI service not available"}), 500

        # Extract data to get categories
        df = ai_demo.extract_real_data()
        if df.empty:
            return jsonify({"categories": []}), 200

        categories = ai_demo.get_available_categories(df)

        # Get category statistics
        category_stats = []
        category_column = (
            "CategorieName" if "CategorieName" in df.columns else "category"
        )

        for category in categories:
            category_data = df[df[category_column] == category]
            stats = {
                "name": category,
                "product_count": len(category_data),
                "promotion_needed": (
                    int(category_data["should_promote"].sum())
                    if "should_promote" in category_data.columns
                    else 0
                ),
                "avg_stock_coverage": (
                    float(category_data["stock_coverage_days"].mean())
                    if "stock_coverage_days" in category_data.columns
                    else 0
                ),
                "avg_price": (
                    float(category_data["current_price"].mean())
                    if "current_price" in category_data.columns
                    else 0
                ),
            }
            category_stats.append(stats)

        return jsonify(
            {
                "categories": category_stats,
                "total_products": len(df),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to retrieve categories"}), 500


@app.route("/api/promotions/generate", methods=["POST"])
def generate_promotions():
    """Generate promotion recommendations for a specific category"""
    try:
        global ai_demo
        if ai_demo is None:
            if not initialize_ai_demo():
                return jsonify({"error": "AI service not available"}), 500

        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        category = data.get("category")
        start_date = data.get("start_date", datetime.now().strftime("%Y-%m-%d"))

        if not category:
            return jsonify({"error": "Category is required"}), 400

        logger.info(
            f"Generating promotions for category: {category}, start_date: {start_date}"
        )

        # Extract data
        df = ai_demo.extract_real_data()
        if df.empty:
            return jsonify({"error": "No product data available"}), 500

        # Train models if not already trained
        if ai_demo.promotion_classifier is None:
            logger.info("Training AI models...")
            ai_demo.train_models(df)

        # Analyze category promotion
        category_data, recommendations_df = ai_demo.analyze_category_promotion(
            df, category, start_date
        )

        if recommendations_df.empty:
            return jsonify(
                {
                    "success": True,
                    "message": f'No products in category "{category}" need promotion at this time',
                    "category": category,
                    "start_date": start_date,
                    "recommendations": [],
                    "summary": {
                        "total_products": len(category_data),
                        "products_to_promote": 0,
                        "expected_revenue_increase": 0,
                        "avg_discount": 0,
                    },
                }
            )

        # Convert recommendations to JSON format
        recommendations = []
        total_expected_revenue = 0

        for _, rec in recommendations_df.iterrows():
            current_price = float(rec.get("current_price", 0))
            suggested_discount = float(rec.get("suggested_discount", 0))

            # Calculate discounted price
            discounted_price = current_price * (1 - suggested_discount / 100)

            recommendation = {
                "product_name": rec.get("product_name", "Unknown"),
                "code_article": rec.get("code_article", "N/A"),
                "current_price": current_price,
                "current_stock": int(rec.get("current_stock", 0)),
                "stock_coverage_days": float(rec.get("stock_coverage_days", 0)),
                "rotation": float(rec.get("rotation", 0)),
                "suggested_discount": suggested_discount,
                "discounted_price": round(discounted_price, 2),
                "predicted_sales_lift": float(rec.get("predicted_sales_lift", 0)),
                "confidence": float(rec.get("confidence", 0)),
                "expected_revenue_increase": float(
                    rec.get("expected_revenue_increase", 0)
                ),
                "reasoning": rec.get("reasoning", ""),
                "start_date": rec.get("start_date", start_date),
                "end_date": rec.get("end_date", start_date),
                "duration_days": int(rec.get("duration_days", 14)),
            }
            recommendations.append(recommendation)
            total_expected_revenue += recommendation["expected_revenue_increase"]

        # Calculate summary statistics
        avg_discount = (
            recommendations_df["suggested_discount"].mean()
            if len(recommendations_df) > 0
            else 0
        )

        summary = {
            "category": category,
            "total_products": len(category_data),
            "products_to_promote": len(recommendations),
            "avg_discount": float(avg_discount),
            "total_expected_revenue_increase": float(total_expected_revenue),
            "start_date": start_date,
            "end_date": (
                recommendations_df.iloc[0]["end_date"]
                if len(recommendations_df) > 0
                else start_date
            ),
            "duration_days": (
                int(recommendations_df.iloc[0]["duration_days"])
                if len(recommendations_df) > 0
                else 14
            ),
        }

        response = {
            "success": True,
            "message": f"Generated {len(recommendations)} promotion recommendations for {category}",
            "recommendations": recommendations,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"✅ Successfully generated {len(recommendations)} recommendations for {category}"
        )
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error generating promotions: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to generate promotions: {str(e)}"}), 500


@app.route("/api/promotions/analyze-product", methods=["POST"])
def analyze_single_product():
    """Analyze a single product for promotion recommendation"""
    try:
        global ai_demo
        if ai_demo is None:
            if not initialize_ai_demo():
                return jsonify({"error": "AI service not available"}), 500

        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No product data provided"}), 400

        # Validate required fields
        required_fields = ["current_price", "current_stock", "total_sales_90d"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Set default values for missing optional fields
        product_data = {
            "current_price": float(data["current_price"]),
            "current_stock": int(data["current_stock"]),
            "total_sales_90d": int(data["total_sales_90d"]),
            "rotation": float(data.get("rotation", 0.3)),
            "sell_through_rate": float(data.get("sell_through_rate", 0.5)),
            "stock_coverage_days": float(data.get("stock_coverage_days", 30)),
            "inventory_turnover": float(data.get("inventory_turnover", 2.0)),
            "sales_trend": float(data.get("sales_trend", 0.0)),
            "profit_margin": float(data.get("profit_margin", 0.4)),
            "days_since_last_promo": int(data.get("days_since_last_promo", 90)),
            "last_promo_discount": float(data.get("last_promo_discount", 0.0)),
            "promo_count_6months": int(data.get("promo_count_6months", 1)),
        }

        # Train models if needed (using sample data since we don't have full dataset)
        if ai_demo.promotion_classifier is None:
            logger.info("Training AI models with sample data...")
            sample_df = ai_demo.generate_sample_data(50)
            ai_demo.train_models(sample_df)

        # Get prediction
        recommendation = ai_demo.predict_promotion(product_data)

        # Calculate expected revenue impact
        monthly_baseline_revenue = product_data["current_price"] * (
            product_data["total_sales_90d"] / 3
        )
        expected_revenue_increase = (
            monthly_baseline_revenue * recommendation["predicted_sales_lift"]
        )

        response = {
            "success": True,
            "product_data": product_data,
            "recommendation": {
                "should_promote": recommendation["should_promote"],
                "confidence_score": recommendation["confidence_score"],
                "optimal_discount": recommendation["optimal_discount"],
                "predicted_sales_lift": recommendation["predicted_sales_lift"],
                "expected_revenue_increase": float(expected_revenue_increase),
                "key_factors": recommendation["key_factors"],
                "risk_assessment": recommendation["risk_assessment"],
                "recommendation_reason": recommendation["recommendation_reason"],
            },
            "timestamp": datetime.now().isoformat(),
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error analyzing product: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to analyze product: {str(e)}"}), 500


@app.route("/api/promotions/model-status", methods=["GET"])
def get_model_status():
    """Get the status of AI models"""
    try:
        global ai_demo
        if ai_demo is None:
            return jsonify(
                {
                    "models_loaded": False,
                    "promotion_classifier": False,
                    "discount_regressor": False,
                    "impact_regressor": False,
                }
            )

        status = {
            "models_loaded": ai_demo.promotion_classifier is not None,
            "promotion_classifier": ai_demo.promotion_classifier is not None,
            "discount_regressor": ai_demo.discount_regressor is not None,
            "impact_regressor": ai_demo.impact_regressor is not None,
            "database_connected": ai_demo.db.test_connection(),
            "timestamp": datetime.now().isoformat(),
        }

        return jsonify(status)

    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({"error": "Failed to get model status"}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("🚀 Starting AI Promotion Flask API")
    print("=" * 50)

    # Initialize AI demo
    if initialize_ai_demo():
        print("✅ AI Promotion service initialized")
    else:
        print("⚠️  AI Promotion service failed to initialize, but API will start anyway")

    print("\n📡 Available endpoints:")
    print("  GET  /api/health                    - Health check")
    print("  GET  /api/promotions/categories     - Get product categories")
    print("  POST /api/promotions/generate       - Generate promotions")
    print("  POST /api/promotions/analyze-product - Analyze single product")
    print("  GET  /api/promotions/model-status   - Get model status")

    print(f"\n🌐 API will be available at: http://localhost:5000")
    print("🔄 CORS enabled for Angular frontend")
    print("=" * 50)

    # Run Flask app
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
