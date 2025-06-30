from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
import pyodbc
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# SQL Server Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = (
    'mssql+pyodbc://DESKTOP-S22JEMV\\SQLEXPRESS/SmartPromoDb_v2024'
    '?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'promotion-secret-key'

db = SQLAlchemy(app)

@dataclass
class ProductAnalysis:
    """Data class for product analysis results"""
    product_id: int
    product_name: str
    category_id: int
    current_price: float
    avg_monthly_sales: float
    total_revenue: float
    profit_margin: float
    stock_level: int
    sales_trend: str  # 'increasing', 'decreasing', 'stable'
    needs_promotion: bool
    recommended_discount: float
    projected_increase: float

@dataclass
class PromotionResult:
    """Data class for promotion calculation results"""
    product_id: int
    discount_percentage: float
    new_price: float
    projected_sales_increase: float
    projected_revenue_increase: float
    promotion_start_date: datetime
    promotion_end_date: datetime
    created_at: datetime

class PromotionModel:
    """Smart Promotion Model for maximizing revenue and CA"""
    
    def __init__(self, db_session):
        self.db = db_session
        self.min_profit_margin = 0.10  # Minimum 10% profit margin
        self.max_discount = 0.30  # Maximum 30% discount
        self.target_sales_increase = 0.20  # Target 20% sales increase
    
    def get_database_tables(self) -> List[str]:
        """Get all tables from the database"""
        try:
            query = text("""
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE = 'BASE TABLE' 
                ORDER BY TABLE_NAME
            """)
            result = self.db.execute(query)
            tables = [row[0] for row in result.fetchall()]
            logger.info(f"Found {len(tables)} tables in database")
            return tables
        except Exception as e:
            logger.error(f"Error getting database tables: {e}")
            return []
    
    def analyze_database_structure(self) -> Dict[str, List[str]]:
        """Analyze database structure and identify key tables"""
        try:
            tables = self.get_database_tables()
            structure = {}
            
            for table in tables:
                try:
                    query = text(f"""
                        SELECT COLUMN_NAME, DATA_TYPE 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_NAME = '{table}'
                        ORDER BY ORDINAL_POSITION
                    """)
                    result = self.db.execute(query)
                    columns = [f"{row[0]} ({row[1]})" for row in result.fetchall()]
                    structure[table] = columns
                except Exception as e:
                    logger.warning(f"Could not analyze table {table}: {e}")
                    structure[table] = ["Error analyzing table"]
            
            return structure
        except Exception as e:
            logger.error(f"Error analyzing database structure: {e}")
            return {}
    
    def get_products_by_category(self, category_id: int) -> List[Dict]:
        """Get all products under a specific category"""
        try:
            query = text("""
                SELECT 
                    a.Id as product_id,
                    a.Libelle as product_name,
                    a.Prix_Vente_TND as price,
                    a.IdCategorie as category_id,
                    c.Nom as category_name,
                    COALESCE(s.QuantitePhysique, 0) as stock_quantity
                FROM Articles a
                LEFT JOIN Categories c ON a.IdCategorie = c.IdCategorie
                LEFT JOIN Stocks s ON a.Id = s.ArticleId
                WHERE a.IdCategorie = :category_id
                ORDER BY a.Libelle
            """)
            result = self.db.execute(query, {"category_id": str(category_id)})
            products = []
            
            for row in result.fetchall():
                products.append({
                    'product_id': row[0],
                    'product_name': row[1],
                    'price': float(row[2]) if row[2] else 0.0,
                    'category_id': row[3],
                    'category_name': row[4],
                    'stock_quantity': int(row[5]) if row[5] else 0
                })
            
            logger.info(f"Found {len(products)} products in category {category_id}")
            return products
            
        except Exception as e:
            logger.error(f"Error getting products by category: {e}")
            return []
    
    def get_sales_data(self, product_id: int, days: int = 90) -> Dict:
        """Get sales data for a product over specified days"""
        try:
            # Get sales data from Ventes table through Stocks
            query = text("""
                SELECT 
                    COUNT(*) as sales_count,
                    COALESCE(SUM(v.QuantiteFacturee), 0) as total_quantity,
                    COALESCE(SUM(v.QuantiteFacturee * v.Prix_Vente_TND), 0) as total_revenue,
                    COALESCE(AVG(v.Prix_Vente_TND), 0) as avg_price
                FROM Ventes v
                INNER JOIN Stocks s ON v.StockId = s.Id
                WHERE s.ArticleId = :product_id 
                AND v.Date >= DATEADD(day, -:days, GETDATE())
            """)
            result = self.db.execute(query, {"product_id": product_id, "days": days})
            row = result.fetchone()
            
            if row:
                return {
                    'sales_count': int(row[0]),
                    'total_quantity': int(row[1]) if row[1] else 0,
                    'total_revenue': float(row[2]) if row[2] else 0.0,
                    'avg_price': float(row[3]) if row[3] else 0.0,
                    'avg_monthly_sales': (int(row[1]) if row[1] else 0) * 30 / days
                }
            else:
                return {
                    'sales_count': 0,
                    'total_quantity': 0,
                    'total_revenue': 0.0,
                    'avg_price': 0.0,
                    'avg_monthly_sales': 0.0
                }
                
        except Exception as e:
            logger.error(f"Error getting sales data for product {product_id}: {e}")
            return {
                'sales_count': 0,
                'total_quantity': 0,
                'total_revenue': 0.0,
                'avg_price': 0.0,
                'avg_monthly_sales': 0.0
            }
    
    def calculate_sales_trend(self, product_id: int) -> str:
        """Calculate sales trend: increasing, decreasing, or stable"""
        try:
            # Compare last 30 days vs previous 30 days
            query = text("""
                SELECT 
                    SUM(CASE WHEN v.Date >= DATEADD(day, -30, GETDATE()) THEN v.QuantiteFacturee ELSE 0 END) as recent_sales,
                    SUM(CASE WHEN v.Date >= DATEADD(day, -60, GETDATE()) AND v.Date < DATEADD(day, -30, GETDATE()) THEN v.QuantiteFacturee ELSE 0 END) as previous_sales
                FROM Ventes v
                INNER JOIN Stocks s ON v.StockId = s.Id
                WHERE s.ArticleId = :product_id
                AND v.Date >= DATEADD(day, -60, GETDATE())
            """)
            result = self.db.execute(query, {"product_id": product_id})
            row = result.fetchone()
            
            if row:
                recent_sales = int(row[0]) if row[0] else 0
                previous_sales = int(row[1]) if row[1] else 0
                
                if previous_sales == 0:
                    return 'stable' if recent_sales == 0 else 'increasing'
                
                change_ratio = (recent_sales - previous_sales) / previous_sales
                
                if change_ratio > 0.1:
                    return 'increasing'
                elif change_ratio < -0.1:
                    return 'decreasing'
                else:
                    return 'stable'
            
            return 'stable'
            
        except Exception as e:
            logger.error(f"Error calculating sales trend for product {product_id}: {e}")
            return 'stable'
    
    def calculate_profit_margin(self, selling_price: float, product_id: int) -> float:
        """Calculate profit margin (assuming cost is 70% of selling price)"""
        try:
            # In a real scenario, you would have a cost table
            # For now, assuming cost is 70% of selling price
            estimated_cost = selling_price * 0.70
            profit = selling_price - estimated_cost
            return profit / selling_price if selling_price > 0 else 0.0
        except:
            return 0.0
    
    def needs_promotion(self, analysis: Dict) -> bool:
        """Determine if a product needs promotion based on analysis"""
        # Product needs promotion if:
        # 1. Sales are decreasing
        # 2. Sales are stable but low
        # 3. High stock levels
        # 4. Low recent sales
        
        if analysis['sales_trend'] == 'decreasing':
            return True
        
        if analysis['sales_trend'] == 'stable' and analysis['avg_monthly_sales'] < 10:
            return True
        
        if analysis['stock_quantity'] > 50 and analysis['avg_monthly_sales'] < 20:
            return True
        
        if analysis['avg_monthly_sales'] < 5:
            return True
        
        return False
    
    def calculate_optimal_discount(self, analysis: Dict) -> float:
        """Calculate optimal discount percentage based on product analysis"""
        base_discount = 0.05  # 5% base discount
        
        # Increase discount based on various factors
        if analysis['sales_trend'] == 'decreasing':
            base_discount += 0.10  # +10% for decreasing sales
        
        if analysis['avg_monthly_sales'] < 5:
            base_discount += 0.08  # +8% for very low sales
        elif analysis['avg_monthly_sales'] < 15:
            base_discount += 0.05  # +5% for low sales
        
        if analysis['stock_quantity'] > 100:
            base_discount += 0.07  # +7% for high stock
        elif analysis['stock_quantity'] > 50:
            base_discount += 0.03  # +3% for medium-high stock
        
        # Ensure profit margin is maintained
        profit_margin = self.calculate_profit_margin(analysis['price'], analysis['product_id'])
        max_allowed_discount = max(0.05, profit_margin - self.min_profit_margin)
        
        # Cap the discount
        optimal_discount = min(base_discount, max_allowed_discount, self.max_discount)
        
        return round(optimal_discount, 2)
    
    def project_sales_increase(self, discount: float) -> float:
        """Project sales increase based on discount percentage"""
        # Elasticity model: sales increase = discount * elasticity_factor
        # Higher discounts generally yield diminishing returns
        elasticity_factor = 2.5 - (discount * 2)  # Diminishing returns
        projected_increase = discount * elasticity_factor
        
        return min(projected_increase, 1.0)  # Cap at 100% increase
    
    def analyze_product(self, product: Dict) -> ProductAnalysis:
        """Comprehensive analysis of a single product"""
        try:
            # Get sales data
            sales_data = self.get_sales_data(product['product_id'])
            
            # Calculate sales trend
            sales_trend = self.calculate_sales_trend(product['product_id'])
            
            # Calculate profit margin
            profit_margin = self.calculate_profit_margin(product['price'], product['product_id'])
            
            # Combine all data
            analysis_data = {
                **product,
                **sales_data,
                'sales_trend': sales_trend,
                'profit_margin': profit_margin
            }
            
            # Determine if promotion is needed
            needs_promo = self.needs_promotion(analysis_data)
            
            # Calculate optimal discount if promotion is needed
            discount = 0.0
            projected_increase = 0.0
            
            if needs_promo:
                discount = self.calculate_optimal_discount(analysis_data)
                projected_increase = self.project_sales_increase(discount)
            
            return ProductAnalysis(
                product_id=product['product_id'],
                product_name=product['product_name'],
                category_id=product['category_id'],
                current_price=product['price'],
                avg_monthly_sales=sales_data['avg_monthly_sales'],
                total_revenue=sales_data['total_revenue'],
                profit_margin=profit_margin,
                stock_level=product['stock_quantity'],
                sales_trend=sales_trend,
                needs_promotion=needs_promo,
                recommended_discount=discount,
                projected_increase=projected_increase
            )
            
        except Exception as e:
            logger.error(f"Error analyzing product {product['product_id']}: {e}")
            return None
    
    def generate_promotions_for_category(self, category_id: int) -> List[PromotionResult]:
        """Generate promotions for all products in a category"""
        try:
            # Get all products in category
            products = self.get_products_by_category(category_id)
            
            if not products:
                logger.warning(f"No products found in category {category_id}")
                return []
            
            promotions = []
            
            for product in products:
                # Analyze product
                analysis = self.analyze_product(product)
                
                if analysis and analysis.needs_promotion:
                    # Create promotion
                    promotion = PromotionResult(
                        product_id=analysis.product_id,
                        discount_percentage=analysis.recommended_discount,
                        new_price=analysis.current_price * (1 - analysis.recommended_discount),
                        projected_sales_increase=analysis.projected_increase,
                        projected_revenue_increase=analysis.total_revenue * analysis.projected_increase,
                        promotion_start_date=datetime.now(),
                        promotion_end_date=datetime.now() + timedelta(days=30),
                        created_at=datetime.now()
                    )
                    
                    promotions.append(promotion)
                    logger.info(f"Generated promotion for product {analysis.product_id}: {analysis.recommended_discount*100:.1f}% discount")
            
            logger.info(f"Generated {len(promotions)} promotions for category {category_id}")
            return promotions
            
        except Exception as e:
            logger.error(f"Error generating promotions for category {category_id}: {e}")
            return []
    
    def save_promotion_to_db(self, promotion: PromotionResult) -> bool:
        """Save promotion to database"""
        try:
            # Get the CodeArticle for the product
            article_query = text("SELECT CodeArticle FROM Articles WHERE Id = :product_id")
            result = self.db.execute(article_query, {"product_id": promotion.product_id})
            article_row = result.fetchone()
            
            if not article_row:
                logger.error(f"Could not find article with ID {promotion.product_id}")
                return False
            
            code_article = article_row[0]
            
            # Insert promotion using the existing table structure
            insert_query = text("""
                INSERT INTO Promotions 
                (DateFin, TauxReduction, CodeArticle, Prix_Vente_TND_Avant, Prix_Vente_TND_Apres, 
                 IsAccepted, DateCreation, ExpectedVolumeImpact, ExpectedRevenueImpact)
                VALUES (:date_fin, :taux_reduction, :code_article, :prix_avant, :prix_apres,
                        :is_accepted, :date_creation, :expected_volume_impact, :expected_revenue_impact)
            """)
            
            self.db.execute(insert_query, {
                "date_fin": promotion.promotion_end_date,
                "taux_reduction": promotion.discount_percentage,
                "code_article": code_article,
                "prix_avant": promotion.new_price / (1 - promotion.discount_percentage),  # Calculate original price
                "prix_apres": promotion.new_price,
                "is_accepted": False,  # Initially not accepted
                "date_creation": promotion.created_at,
                "expected_volume_impact": promotion.projected_sales_increase,
                "expected_revenue_impact": promotion.projected_revenue_increase
            })
            
            self.db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error saving promotion to database: {e}")
            self.db.rollback()
            return False

# Initialize the promotion model
promotion_model = PromotionModel(db.session)

# API Endpoints
@app.route('/')
def home():
    return jsonify({
        "message": "Smart Promotion Model API",
        "endpoints": [
            "GET /database-structure - Analyze database structure",
            "GET /categories - List all categories",
            "POST /generate-promotions - Generate promotions for category",
            "GET /promotions - List all active promotions"
        ]
    })

@app.route('/database-structure')
def get_database_structure():
    """Analyze and return database structure"""
    try:
        structure = promotion_model.analyze_database_structure()
        return jsonify({
            "status": "success",
            "tables_count": len(structure),
            "tables": structure
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/categories')
def get_categories():
    """Get all categories"""
    try:
        query = text("SELECT IdCategorie, Nom FROM Categories ORDER BY Nom")
        result = db.session.execute(query)
        categories = [{"id": row[0], "name": row[1]} for row in result.fetchall()]
        
        return jsonify({
            "status": "success",
            "categories": categories
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/generate-promotions', methods=['POST'])
def generate_promotions():
    """Generate promotions for a specific category"""
    try:
        data = request.get_json()
        category_id = data.get('category_id')
        
        if not category_id:
            return jsonify({
                "status": "error",
                "message": "category_id is required"
            }), 400
        
        # Generate promotions
        promotions = promotion_model.generate_promotions_for_category(category_id)
        
        # Save promotions to database
        saved_count = 0
        for promotion in promotions:
            if promotion_model.save_promotion_to_db(promotion):
                saved_count += 1
        
        # Format response
        promotion_data = []
        for promo in promotions:
            promotion_data.append({
                "product_id": promo.product_id,
                "discount_percentage": f"{promo.discount_percentage*100:.1f}%",
                "new_price": f"{promo.new_price:.2f}",
                "projected_sales_increase": f"{promo.projected_sales_increase*100:.1f}%",
                "projected_revenue_increase": f"{promo.projected_revenue_increase:.2f}",
                "start_date": promo.promotion_start_date.strftime("%Y-%m-%d"),
                "end_date": promo.promotion_end_date.strftime("%Y-%m-%d")
            })
        
        return jsonify({
            "status": "success",
            "category_id": category_id,
            "promotions_generated": len(promotions),
            "promotions_saved": saved_count,
            "promotions": promotion_data
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/promotions')
def get_active_promotions():
    """Get all active promotions"""
    try:
        query = text("""
            SELECT p.*, a.Libelle as product_name, c.Nom as category_name
            FROM Promotions p
            LEFT JOIN Articles a ON p.CodeArticle = a.CodeArticle
            LEFT JOIN Categories c ON a.IdCategorie = c.IdCategorie
            WHERE p.IsAccepted = 1 AND p.DateFin > GETDATE()
            ORDER BY p.DateCreation DESC
        """)
        result = db.session.execute(query)
        
        promotions = []
        for row in result.fetchall():
            promotions.append({
                "id": row[0],
                "code_article": row[3],
                "product_name": row[16] if len(row) > 16 else "Unknown",
                "category_name": row[17] if len(row) > 17 else "Unknown",
                "discount_percentage": f"{float(row[2])*100:.1f}%",
                "price_before": f"{float(row[4]):.2f}",
                "price_after": f"{float(row[5]):.2f}",
                "end_date": row[1].strftime("%Y-%m-%d"),
                "created_at": row[7].strftime("%Y-%m-%d %H:%M")
            })
        
        return jsonify({
            "status": "success",
            "promotions_count": len(promotions),
            "promotions": promotions
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/analyze-category/<int:category_id>')
def analyze_category(category_id):
    """Analyze products in a category without generating promotions"""
    try:
        products = promotion_model.get_products_by_category(category_id)
        
        analysis_results = []
        for product in products:
            analysis = promotion_model.analyze_product(product)
            if analysis:
                analysis_results.append({
                    "product_id": analysis.product_id,
                    "product_name": analysis.product_name,
                    "current_price": f"{analysis.current_price:.2f}",
                    "avg_monthly_sales": f"{analysis.avg_monthly_sales:.1f}",
                    "total_revenue": f"{analysis.total_revenue:.2f}",
                    "profit_margin": f"{analysis.profit_margin*100:.1f}%",
                    "stock_level": analysis.stock_level,
                    "sales_trend": analysis.sales_trend,
                    "needs_promotion": analysis.needs_promotion,
                    "recommended_discount": f"{analysis.recommended_discount*100:.1f}%",
                    "projected_increase": f"{analysis.projected_increase*100:.1f}%"
                })
        
        return jsonify({
            "status": "success",
            "category_id": category_id,
            "products_analyzed": len(analysis_results),
            "products_needing_promotion": sum(1 for p in analysis_results if p["needs_promotion"]),
            "analysis": analysis_results
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/debug-category/<int:category_id>')
def debug_category(category_id):
    """Debug category analysis with detailed information"""
    try:
        products = promotion_model.get_products_by_category(category_id)
        
        debug_info = {
            "products_found": len(products),
            "products_details": [],
            "promotion_criteria": {
                "needs_promotion_if": [
                    "Sales trend is decreasing",
                    "Sales are stable but < 10 units/month", 
                    "Stock > 50 and sales < 20/month",
                    "Average monthly sales < 5"
                ]
            }
        }
        
        for product in products:
            # Get detailed analysis
            sales_data = promotion_model.get_sales_data(product['product_id'])
            sales_trend = promotion_model.calculate_sales_trend(product['product_id'])
            
            # Check each promotion criteria
            criteria_met = []
            if sales_trend == 'decreasing':
                criteria_met.append("Sales decreasing")
            if sales_trend == 'stable' and sales_data['avg_monthly_sales'] < 10:
                criteria_met.append(f"Stable but low sales ({sales_data['avg_monthly_sales']:.1f}/month)")
            if product['stock_quantity'] > 50 and sales_data['avg_monthly_sales'] < 20:
                criteria_met.append(f"High stock ({product['stock_quantity']}) with low sales ({sales_data['avg_monthly_sales']:.1f}/month)")
            if sales_data['avg_monthly_sales'] < 5:
                criteria_met.append(f"Very low sales ({sales_data['avg_monthly_sales']:.1f}/month)")
            
            needs_promo = len(criteria_met) > 0
            
            debug_info["products_details"].append({
                "product_id": product['product_id'],
                "product_name": product['product_name'],
                "price": product['price'],
                "stock": product['stock_quantity'],
                "sales_data": sales_data,
                "sales_trend": sales_trend,
                "needs_promotion": needs_promo,
                "criteria_met": criteria_met,
                "raw_product_data": product
            })
        
        return jsonify({
            "status": "success",
            "debug_info": debug_info
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/test-sales-data/<int:product_id>')
def test_sales_data(product_id):
    """Test sales data retrieval for a specific product"""
    try:
        sales_data = promotion_model.get_sales_data(product_id)
        sales_trend = promotion_model.calculate_sales_trend(product_id)
        
        # Also get raw sales records
        raw_query = text("""
            SELECT TOP 10 
                v.Date, v.QuantiteFacturee, v.Prix_Vente_TND, 
                v.QuantiteFacturee * v.Prix_Vente_TND as total
            FROM Ventes v
            INNER JOIN Stocks s ON v.StockId = s.Id
            WHERE s.ArticleId = :product_id
            ORDER BY v.Date DESC
        """)
        result = db.session.execute(raw_query, {"product_id": product_id})
        raw_sales = []
        for row in result.fetchall():
            raw_sales.append({
                "date": row[0].strftime("%Y-%m-%d") if row[0] else "Unknown",
                "quantity": int(row[1]) if row[1] else 0,
                "unit_price": float(row[2]) if row[2] else 0.0,
                "total": float(row[3]) if row[3] else 0.0
            })
        
        return jsonify({
            "status": "success",
            "product_id": product_id,
            "sales_data": sales_data,
            "sales_trend": sales_trend,
            "recent_sales_records": raw_sales,
            "has_sales_data": len(raw_sales) > 0
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/force-promotion/<int:category_id>')
def force_promotion(category_id):
    """Force generate promotions with relaxed criteria for testing"""
    try:
        products = promotion_model.get_products_by_category(category_id)
        
        if not products:
            return jsonify({
                "status": "error",
                "message": f"No products found in category {category_id}"
            }), 404
        
        forced_promotions = []
        
        for product in products:
            # Force create promotion for testing (relaxed criteria)
            sales_data = promotion_model.get_sales_data(product['product_id'])
            
            # Create promotion with default 10% discount
            promotion = {
                "product_id": product['product_id'],
                "product_name": product['product_name'],
                "current_price": product['price'],
                "discount_percentage": 10.0,
                "new_price": product['price'] * 0.9,
                "sales_data": sales_data,
                "stock_level": product['stock_quantity'],
                "reason": "Forced promotion for testing"
            }
            
            forced_promotions.append(promotion)
        
        return jsonify({
            "status": "success",
            "category_id": category_id,
            "forced_promotions": len(forced_promotions),
            "promotions": forced_promotions
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Smart Promotion Model API")
    print("=" * 60)
    print("Database: SmartPromoDb_v2024")
    print("Features:")
    print("- Automatic database analysis")
    print("- Smart promotion calculation")
    print("- Category-based promotion generation")
    print("- Revenue optimization")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
