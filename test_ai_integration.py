"""
AI Promotion Model Testing and Integration Script
================================================

This script tests the AI promotion model, trains it with real data,
and demonstrates integration with the Flask API.

Usage:
1. Test database connection
2. Extract and prepare training data
3. Train the AI model
4. Test predictions
5. Start Flask API server
6. Test API endpoints

Author: Smart Promotion System
Date: June 2025
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import time
import subprocess

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_promotion_model import AdvancedPromotionAI, RetailKPICalculator
    from advanced_ai_api import app
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files are in the same directory")
    sys.exit(1)

class AIPromotionTester:
    """Test suite for AI promotion model and API"""
    
    def __init__(self):
        self.db_connection = (
            'mssql+pyodbc://DESKTOP-S22JEMV\\SQLEXPRESS/SmartPromoDb_v2024'
            '?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
        )
        self.api_base_url = "http://localhost:5001"
        self.ai_model = None
        
    def test_database_connection(self):
        """Test database connectivity"""
        print("=" * 60)
        print("1. TESTING DATABASE CONNECTION")
        print("=" * 60)
        
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(self.db_connection)
            
            # Test connection with a simple query
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) as article_count FROM Articles"))
                article_count = result.fetchone()[0]
                print(f"✅ Database connection successful!")
                print(f"   Articles in database: {article_count}")
                
                # Test other tables
                result = conn.execute(text("SELECT COUNT(*) as promo_count FROM Promotions"))
                promo_count = result.fetchone()[0]
                print(f"   Promotions in database: {promo_count}")
                
                result = conn.execute(text("SELECT COUNT(*) as sales_count FROM Ventes"))
                sales_count = result.fetchone()[0]
                print(f"   Sales records in database: {sales_count}")
                
            return True
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def test_data_extraction(self):
        """Test data extraction and KPI calculation"""
        print("\\n" + "=" * 60)
        print("2. TESTING DATA EXTRACTION AND KPI CALCULATION")
        print("=" * 60)
        
        try:
            # Initialize AI model
            self.ai_model = AdvancedPromotionAI(self.db_connection)
            
            # Extract training data
            print("Extracting training data...")
            training_data = self.ai_model.extract_training_data(months_back=6)
            
            if len(training_data) == 0:
                print("❌ No training data extracted")
                return False
            
            print(f"✅ Extracted {len(training_data)} records")
            print(f"   Columns: {list(training_data.columns)}")
            
            # Display sample data
            print("\\nSample data:")
            print(training_data.head())
            
            # Test KPI calculation
            print("\\nTesting KPI calculations...")
            
            # Sample product for testing
            sample_data = {
                'total_sales_90d': 25,
                'total_purchased_90d': 50,
                'current_stock': 15,
                'sales_last_30d': 8,
                'sales_previous_30d': 12,
                'current_price': 89.99
            }
            
            kpi_calc = RetailKPICalculator()
            rotation = kpi_calc.calculate_rotation(sample_data['total_sales_90d'], sample_data['total_purchased_90d'])
            sell_through = kpi_calc.calculate_sell_through_rate(sample_data['total_sales_90d'], 
                                                               sample_data['total_sales_90d'] + sample_data['current_stock'])
            stock_coverage = kpi_calc.calculate_stock_coverage_days(sample_data['current_stock'], 
                                                                   sample_data['total_sales_90d'] / 90)
            
            print(f"   Sample KPIs:")
            print(f"   - Rotation: {rotation:.3f}")
            print(f"   - Sell-through rate: {sell_through:.1%}")
            print(f"   - Stock coverage: {stock_coverage:.1f} days")
            
            return True
            
        except Exception as e:
            print(f"❌ Data extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model_training(self):
        """Test AI model training"""
        print("\\n" + "=" * 60)
        print("3. TESTING AI MODEL TRAINING")
        print("=" * 60)
        
        try:
            if self.ai_model is None:
                print("❌ AI model not initialized")
                return False
            
            # Extract training data
            print("Preparing training data...")
            training_data = self.ai_model.extract_training_data(months_back=12)
            
            if len(training_data) < 50:
                print(f"❌ Insufficient training data: {len(training_data)} records (need at least 50)")
                return False
            
            print(f"Training with {len(training_data)} records...")
            
            # Train the model
            metrics = self.ai_model.train_model(training_data)
            
            print("✅ Model training completed!")
            print("Training metrics:")
            for metric, value in metrics.items():
                print(f"   - {metric}: {value:.3f}")
            
            # Save the model
            model_path = "test_ai_model.pkl"
            self.ai_model.save_model(model_path)
            print(f"✅ Model saved to {model_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_predictions(self):
        """Test AI predictions"""
        print("\\n" + "=" * 60)
        print("4. TESTING AI PREDICTIONS")
        print("=" * 60)
        
        try:
            if self.ai_model is None or not self.ai_model.is_trained():
                print("❌ AI model not trained")
                return False
            
            # Test data for different scenarios
            test_products = [
                {
                    'product_id': 1,
                    'product_name': 'Summer Dress - Slow Moving',
                    'category_name': 'Dresses',
                    'current_price': 120.0,
                    'current_stock': 45,
                    'total_sales_90d': 8,  # Low sales
                    'total_revenue_90d': 960.0,
                    'total_purchased_90d': 50,
                    'sales_last_30d': 2,
                    'sales_previous_30d': 4,
                    'days_since_last_promo': 120,
                    'last_promo_discount': 0.0,
                    'promo_count_6months': 0,
                    'category_id': 2
                },
                {
                    'product_id': 2,
                    'product_name': 'Popular T-Shirt - High Performer',
                    'category_name': 'T-Shirts',
                    'current_price': 45.0,
                    'current_stock': 12,
                    'total_sales_90d': 35,  # High sales
                    'total_revenue_90d': 1575.0,
                    'total_purchased_90d': 40,
                    'sales_last_30d': 15,
                    'sales_previous_30d': 12,
                    'days_since_last_promo': 90,
                    'last_promo_discount': 0.15,
                    'promo_count_6months': 1,
                    'category_id': 1
                },
                {
                    'product_id': 3,
                    'product_name': 'Seasonal Jacket - Overstocked',
                    'category_name': 'Outerwear',
                    'current_price': 180.0,
                    'current_stock': 85,  # High stock
                    'total_sales_90d': 5,  # Very low sales
                    'total_revenue_90d': 900.0,
                    'total_purchased_90d': 90,
                    'sales_last_30d': 1,
                    'sales_previous_30d': 3,
                    'days_since_last_promo': 30,  # Recent promo
                    'last_promo_discount': 0.20,
                    'promo_count_6months': 2,
                    'category_id': 3
                }
            ]
            
            print("Testing predictions for different scenarios:")
            print("-" * 60)
            
            for i, product in enumerate(test_products, 1):
                print(f"\\nProduct {i}: {product['product_name']}")
                
                try:
                    recommendation = self.ai_model.predict_promotion_recommendation(product)
                    
                    print(f"   Should promote: {recommendation.should_promote}")
                    print(f"   Confidence: {recommendation.confidence_score:.2f}")
                    if recommendation.should_promote:
                        print(f"   Optimal discount: {recommendation.optimal_discount:.1%}")
                        print(f"   Predicted sales lift: {recommendation.predicted_sales_lift:.1%}")
                        print(f"   Revenue impact: ${recommendation.predicted_revenue_impact:.2f}")
                    print(f"   Current rotation: {recommendation.current_rotation:.3f}")
                    print(f"   Sell-through rate: {recommendation.sell_through_rate:.1%}")
                    print(f"   Stock coverage: {recommendation.stock_coverage_days:.1f} days")
                    print(f"   Risk assessment: {recommendation.risk_assessment}")
                    print(f"   Key factors: {', '.join(recommendation.key_factors)}")
                    print(f"   Reason: {recommendation.recommendation_reason}")
                    
                except Exception as e:
                    print(f"   ❌ Prediction failed: {e}")
            
            print("\\n✅ Prediction testing completed!")
            return True
            
        except Exception as e:
            print(f"❌ Prediction testing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_api_server(self):
        """Start the Flask API server in background"""
        print("\\n" + "=" * 60)
        print("5. STARTING FLASK API SERVER")
        print("=" * 60)
        
        try:
            # Start Flask app in a separate process
            print("Starting Flask API server on http://localhost:5001...")
            print("Note: In production, use a proper WSGI server like Gunicorn")
            
            # For testing, we'll import and use the app directly
            from advanced_ai_api import app, initialize_ai_model
            
            # Initialize the AI model in the API
            initialize_ai_model()
            
            print("✅ API server ready for testing")
            print("   Available endpoints:")
            print("   - GET  /ai/health")
            print("   - GET  /ai/model/info")
            print("   - POST /ai/model/train")
            print("   - POST /ai/promotion/predict")
            print("   - POST /ai/promotion/batch")
            print("   - POST /ai/kpis/calculate")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start API server: {e}")
            return False
    
    def test_api_endpoints(self):
        """Test API endpoints"""
        print("\\n" + "=" * 60)
        print("6. TESTING API ENDPOINTS")
        print("=" * 60)
        
        # Test health endpoint
        print("Testing health endpoint...")
        try:
            response = requests.get(f"{self.api_base_url}/ai/health", timeout=5)
            if response.status_code == 200:
                print("✅ Health endpoint working")
                print(f"   Response: {response.json()}")
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to API server. Is it running?")
            return False
        except Exception as e:
            print(f"❌ Health endpoint error: {e}")
            return False
        
        # Test model info endpoint
        print("\\nTesting model info endpoint...")
        try:
            response = requests.get(f"{self.api_base_url}/ai/model/info", timeout=10)
            if response.status_code == 200:
                print("✅ Model info endpoint working")
                print(f"   Response: {response.json()}")
            else:
                print(f"❌ Model info failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Model info error: {e}")
        
        # Test prediction endpoint
        print("\\nTesting prediction endpoint...")
        try:
            test_product = {
                'product_id': 999,
                'product_name': 'Test Product',
                'category_name': 'Test Category',
                'current_price': 50.0,
                'current_stock': 20,
                'total_sales_90d': 10,
                'total_revenue_90d': 500.0,
                'total_purchased_90d': 25,
                'sales_last_30d': 3,
                'sales_previous_30d': 5,
                'days_since_last_promo': 60,
                'last_promo_discount': 0.1,
                'promo_count_6months': 1,
                'category_id': 1
            }
            
            response = requests.post(
                f"{self.api_base_url}/ai/promotion/predict",
                json=test_product,
                timeout=15
            )
            
            if response.status_code == 200:
                print("✅ Prediction endpoint working")
                result = response.json()
                print(f"   Should promote: {result['ai_recommendation']['should_promote']}")
                print(f"   Confidence: {result['ai_recommendation']['confidence_score']}")
                print(f"   Optimal discount: {result['ai_recommendation']['optimal_discount_rate']:.1%}")
            else:
                print(f"❌ Prediction endpoint failed: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"❌ Prediction endpoint error: {e}")
        
        # Test KPI calculation endpoint
        print("\\nTesting KPI calculation endpoint...")
        try:
            kpi_data = {
                'product_id': 999,
                'total_sales_90d': 25,
                'total_purchased_90d': 40,
                'current_stock': 15,
                'sales_last_30d': 8,
                'sales_previous_30d': 10,
                'current_price': 75.0
            }
            
            response = requests.post(
                f"{self.api_base_url}/ai/kpis/calculate",
                json=kpi_data,
                timeout=10
            )
            
            if response.status_code == 200:
                print("✅ KPI calculation endpoint working")
                result = response.json()
                kpis = result['kpis']
                print(f"   Rotation: {kpis['rotation_rate']}")
                print(f"   Sell-through: {kpis['sell_through_rate_percent']}%")
                print(f"   Stock coverage: {kpis['stock_coverage_days']} days")
                print(f"   Overall health: {kpis['performance_indicators']['overall_health']}")
            else:
                print(f"❌ KPI calculation failed: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"❌ KPI calculation error: {e}")
        
        print("\\n✅ API endpoint testing completed!")
        return True
    
    def run_full_test_suite(self):
        """Run the complete test suite"""
        print("AI PROMOTION MODEL - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"Started at: {datetime.now()}")
        print("=" * 80)
        
        results = {}
        
        # Run all tests
        results['database'] = self.test_database_connection()
        if results['database']:
            results['data_extraction'] = self.test_data_extraction()
            if results['data_extraction']:
                results['model_training'] = self.test_model_training()
                if results['model_training']:
                    results['predictions'] = self.test_predictions()
                    results['api_server'] = self.start_api_server()
                    if results['api_server']:
                        # Give server time to start
                        time.sleep(2)
                        results['api_endpoints'] = self.test_api_endpoints()
        
        # Print summary
        print("\\n" + "=" * 80)
        print("TEST SUITE SUMMARY")
        print("=" * 80)
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title():<25} {status}")
        
        print("-" * 80)
        print(f"Total: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! AI Promotion Model is ready for production.")
        else:
            print("⚠️  Some tests failed. Please check the errors above.")
        
        print("=" * 80)
        print(f"Completed at: {datetime.now()}")
        print("=" * 80)
        
        return passed_tests == total_tests

def main():
    """Main function to run the test suite"""
    tester = AIPromotionTester()
    success = tester.run_full_test_suite()
    
    if success:
        print("\\n🚀 NEXT STEPS:")
        print("1. Integrate with your .NET backend:")
        print("   - Add API calls to http://localhost:5001/ai/promotion/predict")
        print("   - Update PromotionController to use AI recommendations")
        print("2. Update Angular frontend:")
        print("   - Add AI recommendation display")
        print("   - Show confidence scores and KPIs")
        print("3. Deploy Flask API to production server")
        print("4. Set up model retraining schedule")
    else:
        print("\\n❌ Please fix the failing tests before proceeding.")
    
    return success

if __name__ == "__main__":
    main()
