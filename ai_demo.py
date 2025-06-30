"""
AI Promotion Model Demo - Real Database Integration
==================================================

This demo script shows how the AI promotion model works with real data from
your SQL Server database, demonstrating the machine learning capabilities 
for your prêt-à-porter business.

Features Demonstrated:
- Real data extraction from SmartPromoDb_v2024
- Real retail KPI calculations
- AI-powered promotion recommendations  
- Machine learning prediction confidence
- Business rule integration
- Recommendation explanations

Run: python ai_demo.py
"""

import pandas as pd
import numpy as np
import pyodbc
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

class RetailKPICalculator:
    """Calculate retail Key Performance Indicators"""
    
    @staticmethod
    def calculate_rotation(total_sales, quantity_injected):
        """Rotation = Total Sales / Quantity Injected"""
        if quantity_injected == 0:
            return 0.0
        return total_sales / quantity_injected
    
    @staticmethod
    def calculate_sell_through_rate(units_sold, total_available):
        """Sell-through rate = Units Sold / Total Available"""
        if total_available == 0:
            return 0.0
        return units_sold / total_available
    
    @staticmethod
    def calculate_stock_coverage_days(current_stock, daily_sales_rate):
        """Stock coverage = Current Stock / Daily Sales Rate"""
        if daily_sales_rate == 0:
            return 999  # Infinite coverage if no sales
        return current_stock / daily_sales_rate
    
    @staticmethod
    def calculate_inventory_turnover(annual_sales, average_inventory):
        """Inventory turnover = Annual Sales / Average Inventory"""
        if average_inventory == 0:
            return 0.0
        return annual_sales / average_inventory
    
    @staticmethod
    def calculate_gross_margin(selling_price, cost_price):
        """Gross margin = (Selling Price - Cost Price) / Selling Price"""
        if selling_price == 0:
            return 0.0
        return (selling_price - cost_price) / selling_price

class DatabaseConnection:
    """Handle SQL Server database connections and queries"""
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
    
    def get_connection(self):
        """Establish database connection"""
        try:
            conn = pyodbc.connect(self.connection_string)
            return conn
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return None
    
    def execute_query(self, query, params=None):
        """Execute SQL query and return results as DataFrame"""
        conn = self.get_connection()
        if conn is None:
            return pd.DataFrame()
        
        try:
            df = pd.read_sql(query, conn, params=params)
            conn.close()
            return df
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            if conn:
                conn.close()
            return pd.DataFrame()
    
    def test_connection(self):
        """Test database connection"""
        conn = self.get_connection()
        if conn:
            print("✅ Database connection successful!")
            conn.close()
            return True
        else:
            print("❌ Database connection failed!")
            return False

class AIPromotionDemo:
    """Demo of AI Promotion Recommendation System with Real Database Data"""
    
    def __init__(self, connection_string=None):
        self.promotion_classifier = None
        self.discount_regressor = None
        self.impact_regressor = None
        self.scaler = StandardScaler()
        self.kpi_calc = RetailKPICalculator()
        
        # Business thresholds for promotion decisions
        self.min_rotation = 0.3
        self.max_stock_coverage = 60
        self.min_profit_margin = 0.2
        self.min_promo_interval = 30
        self.min_sell_through = 0.3
        self.min_sales_velocity = 0.1
        
        # Database connection
        if connection_string:
            self.db = DatabaseConnection(connection_string)
        else:
            # Default SQL Server connection for SmartPromoDb_v2024
            self.db = DatabaseConnection(
                "Driver={ODBC Driver 17 for SQL Server};"
                "Server=DESKTOP-S22JEMV\\SQLEXPRESS;"
                "Database=SmartPromoDb_v2024;"
                "Trusted_Connection=yes;"
            )
    
    def extract_real_data(self):
        """Extract real data from SQL Server database"""
        print("🔍 Extracting data from SmartPromoDb_v2024...")
        
        # Test connection first
        if not self.db.test_connection():
            print("❌ Cannot connect to database. Using fallback synthetic data.")
            return self.generate_sample_data()
        
        try:
            # Main query to get article data with sales, stock, and promotion history
            main_query = """
            WITH ArticleSales AS (
                SELECT 
                    a.Id as ArticleId,
                    a.CodeArticle,
                    a.Libelle,
                    a.Prix_Vente_TND,
                    a.Prix_Achat_TND,
                    a.FamilleNiv1,
                    a.FamilleNiv2,
                    c.Nom as CategorieName,
                    
                    -- Stock information (sum across all depots)
                    ISNULL(SUM(s.QuantitePhysique), 0) as CurrentStock,
                    ISNULL(SUM(s.StockMin), 0) as StockMin,
                    ISNULL(SUM(s.Valeur_Stock_TND), 0) as StockValue,
                    
                    -- Sales in last 90 days
                    ISNULL(SUM(CASE WHEN v.Date >= DATEADD(day, -90, GETDATE()) 
                               THEN v.QuantiteFacturee ELSE 0 END), 0) as Sales90d,
                    
                    -- Sales in last 30 days  
                    ISNULL(SUM(CASE WHEN v.Date >= DATEADD(day, -30, GETDATE()) 
                               THEN v.QuantiteFacturee ELSE 0 END), 0) as Sales30d,
                    
                    -- Sales in previous 30 days (31-60 days ago)
                    ISNULL(SUM(CASE WHEN v.Date >= DATEADD(day, -60, GETDATE()) 
                                   AND v.Date < DATEADD(day, -30, GETDATE())
                               THEN v.QuantiteFacturee ELSE 0 END), 0) as SalesPrev30d,
                    
                    -- Revenue metrics
                    ISNULL(SUM(CASE WHEN v.Date >= DATEADD(day, -90, GETDATE()) 
                               THEN v.CA_Mensuel_TND ELSE 0 END), 0) as Revenue90d,
                    
                    -- Last sale date
                    MAX(v.Date) as LastSaleDate
                    
                FROM Articles a
                LEFT JOIN Categories c ON a.IdCategorie = c.IdCategorie
                LEFT JOIN Stocks s ON a.Id = s.ArticleId
                LEFT JOIN Ventes v ON s.Id = v.StockId
                
                WHERE a.Prix_Vente_TND > 0  -- Only articles with valid prices
                
                GROUP BY a.Id, a.CodeArticle, a.Libelle, a.Prix_Vente_TND, a.Prix_Achat_TND,
                         a.FamilleNiv1, a.FamilleNiv2, c.Nom
            ),
            PromotionHistory AS (
                SELECT 
                    p.CodeArticle,
                    COUNT(*) as PromoCount6Months,
                    MAX(p.DateFin) as LastPromoDate,
                    AVG(p.TauxReduction) as AvgDiscountRate
                FROM Promotions p
                WHERE p.DateFin >= DATEADD(month, -6, GETDATE())
                GROUP BY p.CodeArticle
            )
            
            SELECT 
                asl.*,
                ISNULL(ph.PromoCount6Months, 0) as PromoCount6Months,
                ph.LastPromoDate,
                ISNULL(ph.AvgDiscountRate, 0) as LastPromoDiscount,
                
                -- Calculate days since last promotion
                CASE 
                    WHEN ph.LastPromoDate IS NOT NULL 
                    THEN DATEDIFF(day, ph.LastPromoDate, GETDATE())
                    ELSE 365 
                END as DaysSinceLastPromo
                
            FROM ArticleSales asl
            LEFT JOIN PromotionHistory ph ON asl.CodeArticle = ph.CodeArticle
            """
            
            print("📊 Executing main data extraction query...")
            df = self.db.execute_query(main_query)
            
            if df.empty:
                print("⚠️ No data returned from database. Using synthetic data.")
                return self.generate_sample_data()
            
            print(f"✅ Extracted {len(df)} articles from database")
            
            # Process and enrich the data
            df = self.process_real_data(df)
            
            return df
            
        except Exception as e:
            print(f"❌ Error extracting data from database: {e}")
            print("🔄 Falling back to synthetic data...")
            return self.generate_sample_data()
    
    def process_real_data(self, df):
        """Process and enrich real database data for AI model"""
        print("🔧 Processing and calculating KPIs...")
        
        # Handle missing values
        df = df.fillna(0)
        
        # Calculate retail KPIs
        df['rotation'] = df.apply(
            lambda row: self.kpi_calc.calculate_rotation(
                row['Sales90d'], max(row['Sales90d'] + row['CurrentStock'], 1)
            ), axis=1
        )
        
        df['sell_through_rate'] = df.apply(
            lambda row: self.kpi_calc.calculate_sell_through_rate(
                row['Sales90d'], max(row['Sales90d'] + row['CurrentStock'], 1)
            ), axis=1
        )
        
        df['stock_coverage_days'] = df.apply(
            lambda row: self.kpi_calc.calculate_stock_coverage_days(
                row['CurrentStock'], max(row['Sales90d'] / 90, 0.1)
            ), axis=1
        )
        
        df['inventory_turnover'] = df.apply(
            lambda row: self.kpi_calc.calculate_inventory_turnover(
                row['Sales90d'] * 4, max(row['CurrentStock'], 1)  # Annualized
            ), axis=1
        )
        
        df['profit_margin'] = df.apply(
            lambda row: self.kpi_calc.calculate_gross_margin(
                row['Prix_Vente_TND'], row['Prix_Achat_TND']
            ), axis=1
        )
        
        # Calculate sales trend
        df['sales_trend'] = df.apply(
            lambda row: (row['Sales30d'] - row['SalesPrev30d']) / max(row['SalesPrev30d'], 1),
            axis=1
        )
        
        # Create promotion need labels (business logic + AI training data)
        df['should_promote'] = self.create_promotion_labels(df)
        
        # Create applied discount for training (simulate some historical promotions)
        df['applied_discount'] = df.apply(
            lambda row: np.random.uniform(0.1, 0.3) if row['should_promote'] and np.random.random() < 0.3 else 0,
            axis=1
        )
        
        # Calculate expected impact (synthetic for training)
        df['expected_volume_increase'] = df.apply(
            lambda row: np.random.uniform(1.2, 2.5) if row['applied_discount'] > 0 else 1.0,
            axis=1
        )
        
        # Rename columns to match the model's expected format
        df = df.rename(columns={
            'Prix_Vente_TND': 'current_price',
            'CurrentStock': 'current_stock',
            'Sales90d': 'total_sales_90d',
            'DaysSinceLastPromo': 'days_since_last_promo',
            'LastPromoDiscount': 'last_promo_discount',
            'PromoCount6Months': 'promo_count_6months'
        })
        
        print(f"✅ Processed {len(df)} articles with KPIs calculated")
        return df
    
    def create_promotion_labels(self, df):
        """Create promotion recommendation labels based on business rules"""
        labels = []
        
        for _, row in df.iterrows():
            should_promote = False
            
            # Rule 1: High stock + Low sales velocity
            if row['CurrentStock'] > 20 and row['Sales90d'] < 10:
                should_promote = True
            
            # Rule 2: Declining sales trend
            if row['sales_trend'] < -0.2:
                should_promote = True
            
            # Rule 3: Long time since last promotion (seasonal refresh)
            if row['DaysSinceLastPromo'] > 120:
                should_promote = True
            
            # Rule 4: Low rotation rate
            if 'rotation' in row and row['rotation'] < 0.3:
                should_promote = True
            
            # Rule 5: High stock coverage (overstock)
            if 'stock_coverage_days' in row and row['stock_coverage_days'] > 60:
                should_promote = True
            
            # Don't promote if recently promoted
            if row['DaysSinceLastPromo'] < 30:
                should_promote = False
            
            # Don't promote if profit margin is too low
            if row['Prix_Vente_TND'] > 0 and row['Prix_Achat_TND'] > 0:
                margin = (row['Prix_Vente_TND'] - row['Prix_Achat_TND']) / row['Prix_Vente_TND']
                if margin < 0.2:  # Less than 20% margin
                    should_promote = False
            
            labels.append(should_promote)
        
        return labels
        
        # Retail business thresholds
        self.min_rotation = 0.5
        self.min_sell_through = 0.3
        self.max_stock_coverage = 90
        
        print("🚀 AI Promotion Model Demo Initialized")
        print("=" * 50)
    
    def generate_sample_data(self, n_products=200):
        """Generate realistic sample data for prêt-à-porter business"""
        print(f"📊 Generating {n_products} sample products...")
        
        np.random.seed(42)  # For reproducible results
        
        # Product categories
        categories = ['Dresses', 'T-Shirts', 'Jeans', 'Jackets', 'Accessories', 'Shoes']
        
        data = []
        for i in range(n_products):
            # Basic product info
            product_id = i + 1
            category = np.random.choice(categories)
            
            # Price ranges by category (prêt-à-porter pricing)
            price_ranges = {
                'Dresses': (80, 250),
                'T-Shirts': (25, 80),
                'Jeans': (60, 150),
                'Jackets': (120, 400),
                'Accessories': (15, 100),
                'Shoes': (70, 300)
            }
            
            min_price, max_price = price_ranges[category]
            current_price = np.random.uniform(min_price, max_price)
            
            # Inventory and sales patterns
            # Some products are high performers, others are slow movers
            performance_type = np.random.choice(['high', 'medium', 'slow'], p=[0.2, 0.5, 0.3])
            
            if performance_type == 'high':
                total_purchased_90d = np.random.randint(30, 100)
                total_sales_90d = np.random.randint(int(total_purchased_90d * 0.6), int(total_purchased_90d * 0.9))
                current_stock = np.random.randint(5, 25)
            elif performance_type == 'medium':
                total_purchased_90d = np.random.randint(20, 60)
                total_sales_90d = np.random.randint(int(total_purchased_90d * 0.3), int(total_purchased_90d * 0.6))
                current_stock = np.random.randint(10, 40)
            else:  # slow
                total_purchased_90d = np.random.randint(15, 50)
                total_sales_90d = np.random.randint(0, int(total_purchased_90d * 0.3))
                current_stock = np.random.randint(20, 80)
            
            # Calculate sales trends
            sales_last_30d = int(total_sales_90d * np.random.uniform(0.2, 0.5))
            sales_previous_30d = int(total_sales_90d * np.random.uniform(0.2, 0.5))
            
            # Promotion history
            days_since_last_promo = np.random.randint(15, 365)
            last_promo_discount = np.random.uniform(0, 0.3) if days_since_last_promo < 180 else 0
            promo_count_6months = np.random.randint(0, 4)
            
            # Calculate KPIs
            rotation = self.kpi_calc.calculate_rotation(total_sales_90d, total_purchased_90d)
            sell_through_rate = self.kpi_calc.calculate_sell_through_rate(
                total_sales_90d, total_sales_90d + current_stock
            )
            stock_coverage_days = self.kpi_calc.calculate_stock_coverage_days(
                current_stock, total_sales_90d / 90
            )
            inventory_turnover = self.kpi_calc.calculate_inventory_turnover(
                total_sales_90d * 4, current_stock  # Annualized
            )
            
            # Sales trend
            sales_trend = 0
            if sales_previous_30d > 0:
                sales_trend = (sales_last_30d - sales_previous_30d) / sales_previous_30d
            
            # Profit margin (typical for clothing retail)
            profit_margin = np.random.uniform(0.35, 0.65)  # 35-65% margin
            
            # Create promotion need label (for supervised learning)
            should_promote = (
                rotation < self.min_rotation or
                sell_through_rate < self.min_sell_through or
                stock_coverage_days > self.max_stock_coverage or
                sales_trend < -0.2
            )
            
            # Simulate promotion outcomes for training
            if should_promote and np.random.random() > 0.3:  # 70% chance of promotion
                applied_discount = np.random.uniform(0.1, 0.3)
                # Simulate promotion effectiveness
                base_lift = applied_discount * np.random.uniform(1.5, 3.0)
                actual_sales_lift = max(0, base_lift + np.random.normal(0, 0.1))
                revenue_impact = actual_sales_lift * current_price * total_sales_90d / 90 * 30
            else:
                applied_discount = 0
                actual_sales_lift = 0
                revenue_impact = 0
            
            data.append({
                'product_id': product_id,
                'product_name': f"{category} Item {product_id}",
                'category': category,
                'current_price': round(current_price, 2),
                'current_stock': current_stock,
                'total_sales_90d': total_sales_90d,
                'total_revenue_90d': round(total_sales_90d * current_price, 2),
                'total_purchased_90d': total_purchased_90d,
                'sales_last_30d': sales_last_30d,
                'sales_previous_30d': sales_previous_30d,
                'days_since_last_promo': days_since_last_promo,
                'last_promo_discount': round(last_promo_discount, 3),
                'promo_count_6months': promo_count_6months,
                'rotation': round(rotation, 3),
                'sell_through_rate': round(sell_through_rate, 3),
                'stock_coverage_days': round(stock_coverage_days, 1),
                'inventory_turnover': round(inventory_turnover, 2),
                'sales_trend': round(sales_trend, 3),
                'profit_margin': round(profit_margin, 3),
                'should_promote': should_promote,
                'applied_discount': applied_discount,
                'actual_sales_lift': round(actual_sales_lift, 3),
                'revenue_impact': round(revenue_impact, 2)
            })
        
        return pd.DataFrame(data)
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        feature_columns = [
            'current_price', 'current_stock', 'total_sales_90d', 'rotation',
            'sell_through_rate', 'stock_coverage_days', 'inventory_turnover',
            'sales_trend', 'profit_margin', 'days_since_last_promo',
            'last_promo_discount', 'promo_count_6months'
        ]
        
        return df[feature_columns].fillna(0)
    
    def train_models(self, df):
        """Train AI models on sample data"""
        print("🤖 Training AI models...")
        
        # Prepare features
        X = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        
        metrics = {}
        
        # 1. Train promotion recommendation classifier
        y_promote = df['should_promote'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_promote, test_size=0.2, random_state=42, stratify=y_promote
        )
        
        self.promotion_classifier = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
        )
        self.promotion_classifier.fit(X_train, y_train)
        
        y_pred = self.promotion_classifier.predict(X_test)
        promotion_accuracy = accuracy_score(y_test, y_pred)
        metrics['promotion_accuracy'] = promotion_accuracy
        
        # 2. Train discount prediction model
        promo_data = df[df['applied_discount'] > 0].copy()
        if len(promo_data) > 20:
            X_promo = self.prepare_features(promo_data)
            X_promo_scaled = self.scaler.transform(X_promo)
            y_discount = promo_data['applied_discount']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_promo_scaled, y_discount, test_size=0.2, random_state=42
            )
            
            self.discount_regressor = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=42
            )
            self.discount_regressor.fit(X_train, y_train)
            
            y_pred = self.discount_regressor.predict(X_test)
            discount_r2 = r2_score(y_test, y_pred)
            metrics['discount_r2'] = discount_r2
        
        # 3. Train sales impact model (if we have historical impact data)
        if 'actual_sales_lift' in df.columns:
            impact_data = df[df['actual_sales_lift'] > 0].copy()
            if len(impact_data) > 20:
                X_impact = self.prepare_features(impact_data)
                X_impact_scaled = self.scaler.transform(X_impact)
                y_impact = impact_data['actual_sales_lift']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_impact_scaled, y_impact, test_size=0.2, random_state=42
                )
                
                self.impact_regressor = RandomForestRegressor(
                    n_estimators=100, max_depth=8, random_state=42
                )
                self.impact_regressor.fit(X_train, y_train)
                
                y_pred = self.impact_regressor.predict(X_test)
                impact_r2 = r2_score(y_test, y_pred)
                metrics['impact_r2'] = impact_r2
        else:
            # Train a basic impact model using expected volume increase
            impact_data = df[df['expected_volume_increase'] > 1.0].copy()
            if len(impact_data) > 20:
                X_impact = self.prepare_features(impact_data)
                X_impact_scaled = self.scaler.transform(X_impact)
                y_impact = impact_data['expected_volume_increase']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_impact_scaled, y_impact, test_size=0.2, random_state=42
                )
                
                self.impact_regressor = RandomForestRegressor(
                    n_estimators=100, max_depth=8, random_state=42
                )
                self.impact_regressor.fit(X_train, y_train)
                
                y_pred = self.impact_regressor.predict(X_test)
                impact_r2 = r2_score(y_test, y_pred)
                metrics['impact_r2'] = impact_r2
        
        print("✅ Model training completed!")
        print(f"   Promotion Classifier Accuracy: {promotion_accuracy:.1%}")
        if 'discount_r2' in metrics:
            print(f"   Discount Predictor R²: {metrics['discount_r2']:.3f}")
        if 'impact_r2' in metrics:
            print(f"   Impact Predictor R²: {metrics['impact_r2']:.3f}")
        
        return metrics
    
    def predict_promotion(self, product_data):
        """Generate AI promotion recommendation"""
        # Prepare features
        features = [
            product_data['current_price'],
            product_data['current_stock'],
            product_data['total_sales_90d'],
            product_data['rotation'],
            product_data['sell_through_rate'],
            product_data['stock_coverage_days'],
            product_data['inventory_turnover'],
            product_data['sales_trend'],
            product_data['profit_margin'],
            product_data['days_since_last_promo'],
            product_data['last_promo_discount'],
            product_data['promo_count_6months']
        ]
        
        features_scaled = self.scaler.transform([features])
        
        # Prediction
        should_promote_proba = self.promotion_classifier.predict_proba(features_scaled)[0]
        should_promote = should_promote_proba[1] > 0.5
        confidence = max(should_promote_proba)
        
        optimal_discount = 0.0
        predicted_sales_lift = 0.0
        
        if should_promote:
            if self.discount_regressor:
                optimal_discount = max(0.05, min(0.3, self.discount_regressor.predict(features_scaled)[0]))
            else:
                optimal_discount = 0.15  # Default
            
            if self.impact_regressor:
                predicted_sales_lift = max(0, self.impact_regressor.predict(features_scaled)[0])
            else:
                predicted_sales_lift = optimal_discount * 1.8  # Estimate
        
        # Get key factors
        key_factors = []
        if product_data['rotation'] < self.min_rotation:
            key_factors.append("Low product rotation")
        if product_data['sell_through_rate'] < self.min_sell_through:
            key_factors.append("Poor sell-through rate")
        if product_data['stock_coverage_days'] > self.max_stock_coverage:
            key_factors.append("Excess inventory")
        if product_data['sales_trend'] < -0.2:
            key_factors.append("Declining sales trend")
        
        # Risk assessment
        if product_data['profit_margin'] < 0.2:
            risk = "HIGH - Low profit margin"
        elif optimal_discount > 0.25:
            risk = "MEDIUM - High discount"
        elif product_data['days_since_last_promo'] < 30:
            risk = "MEDIUM - Recent promotion"
        else:
            risk = "LOW"
        
        # Recommendation reason
        if should_promote:
            reason = f"Promotion recommended due to: {', '.join(key_factors) if key_factors else 'Model prediction'}"
        else:
            reason = "Product performance is satisfactory, no promotion needed"
        
        return {
            'should_promote': should_promote,
            'confidence_score': round(confidence, 3),
            'optimal_discount': round(optimal_discount, 3),
            'predicted_sales_lift': round(predicted_sales_lift, 3),
            'key_factors': key_factors,
            'risk_assessment': risk,
            'recommendation_reason': reason
        }
    
    def run_demo(self):
        """Run the complete AI promotion demo"""
        print("🎯 AI-POWERED PROMOTION OPTIMIZATION DEMO")
        print("For Prêt-à-Porter (Retail Clothing) Business")
        print("=" * 60)
        
        # Generate sample data
        df = self.generate_sample_data(200)
        print(f"✅ Generated data for {len(df)} products")
        
        # Show sample KPIs
        print("\n📈 Sample KPI Overview:")
        print(f"   Average Rotation: {df['rotation'].mean():.2f}")
        print(f"   Average Sell-through: {df['sell_through_rate'].mean():.1%}")
        print(f"   Average Stock Coverage: {df['stock_coverage_days'].mean():.1f} days")
        print(f"   Products needing promotion: {df['should_promote'].sum()}/{len(df)} ({df['should_promote'].mean():.1%})")
        
        # Train models
        print("\\n" + "=" * 60)
        metrics = self.train_models(df)
        
        # Test predictions on sample products
        print("\\n" + "=" * 60)
        print("🔮 AI PREDICTION EXAMPLES")
        print("=" * 60)
        
        # Select different types of products for demo
        test_cases = [
            # High performer - should not promote
            {
                'name': 'Popular T-Shirt (High Performer)',
                'data': df[df['should_promote'] == False].iloc[0] if len(df[df['should_promote'] == False]) > 0 else df.iloc[0]
            },
            # Slow mover - should promote
            {
                'name': 'Slow-Moving Dress (Needs Promotion)',
                'data': df[df['should_promote'] == True].iloc[0] if len(df[df['should_promote'] == True]) > 0 else df.iloc[-1]
            },
            # Medium performer
            {
                'name': 'Medium Performer (Borderline)',
                'data': df.iloc[len(df)//2]
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            product = test_case['data']
            print(f"\\n{i}. {test_case['name']}")
            print("-" * 40)
            print(f"   Product: {product['product_name']}")
            print(f"   Price: ${product['current_price']:.2f}")
            print(f"   Current Stock: {product['current_stock']} units")
            print(f"   Rotation: {product['rotation']:.3f}")
            print(f"   Sell-through: {product['sell_through_rate']:.1%}")
            print(f"   Stock Coverage: {product['stock_coverage_days']:.1f} days")
            
            # Get AI recommendation
            recommendation = self.predict_promotion(product.to_dict())
            
            print(f"\\n   🤖 AI RECOMMENDATION:")
            print(f"   Should Promote: {'YES' if recommendation['should_promote'] else 'NO'}")
            print(f"   Confidence: {recommendation['confidence_score']:.1%}")
            
            if recommendation['should_promote']:
                print(f"   Optimal Discount: {recommendation['optimal_discount']:.1%}")
                print(f"   Expected Sales Lift: {recommendation['predicted_sales_lift']:.1%}")
                expected_revenue = product['current_price'] * product['total_sales_90d'] * recommendation['predicted_sales_lift'] / 3  # Monthly
                print(f"   Expected Revenue Impact: ${expected_revenue:.2f}/month")
            
            print(f"   Risk Level: {recommendation['risk_assessment']}")
            print(f"   Key Factors: {', '.join(recommendation['key_factors']) if recommendation['key_factors'] else 'None'}")
            print(f"   Reason: {recommendation['recommendation_reason']}")
        
        # Summary statistics
        print("\\n" + "=" * 60)
        print("📊 MODEL PERFORMANCE SUMMARY")
        print("=" * 60)
        
        # Test all products
        all_recommendations = []
        for _, product in df.iterrows():
            rec = self.predict_promotion(product.to_dict())
            all_recommendations.append(rec)
        
        recommendations_df = pd.DataFrame(all_recommendations)
        
        total_should_promote = recommendations_df['should_promote'].sum()
        high_confidence = (recommendations_df['confidence_score'] > 0.8).sum()
        avg_discount = recommendations_df[recommendations_df['should_promote']]['optimal_discount'].mean()
        avg_impact = recommendations_df[recommendations_df['should_promote']]['predicted_sales_lift'].mean()
        
        print(f"✅ Processed {len(df)} products")
        print(f"📈 Promotion recommendations: {total_should_promote}/{len(df)} ({total_should_promote/len(df):.1%})")
        print(f"🎯 High confidence predictions: {high_confidence}/{len(df)} ({high_confidence/len(df):.1%})")
        print(f"💰 Average recommended discount: {avg_discount:.1%}")
        print(f"📊 Average predicted sales lift: {avg_impact:.1%}")
        
        print("\\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\\n💡 INTEGRATION NEXT STEPS:")
        print("1. Connect to your real database")
        print("2. Train with historical promotion data")
        print("3. Deploy Flask API for .NET integration")
        print("4. Add to Angular frontend dashboard")
        print("5. Set up automated retraining schedule")
        
        return df, recommendations_df
    
    def run_demo_with_real_data(self):
        """Run the complete AI promotion demo with real database data"""
        print("🎯 AI-POWERED PROMOTION OPTIMIZATION DEMO")
        print("Using Real Data from SmartPromoDb_v2024")
        print("=" * 60)
        
        # Extract real data from database
        df = self.extract_real_data()
        
        if df.empty:
            print("❌ No data available. Cannot proceed with demo.")
            return pd.DataFrame(), pd.DataFrame()
        
        print(f"✅ Loaded data for {len(df)} products from database")
        
        # Show real KPI overview
        print("\n📈 Real Data KPI Overview:")
        print(f"   Average Rotation: {df['rotation'].mean():.2f}")
        print(f"   Average Sell-through: {df['sell_through_rate'].mean():.1%}")
        print(f"   Average Stock Coverage: {df['stock_coverage_days'].mean():.1f} days")
        print(f"   Products needing promotion: {df['should_promote'].sum()}/{len(df)} ({df['should_promote'].mean():.1%})")
        
        # Show category breakdown
        if 'CategorieName' in df.columns:
            print(f"\n📊 Product Categories:")
            cat_counts = df['CategorieName'].value_counts().head(5)
            for cat, count in cat_counts.items():
                print(f"   {cat}: {count} products")
        
        # Train models
        print("\n" + "=" * 60)
        metrics = self.train_models(df)
        
        # Test predictions on real products
        print("\n" + "=" * 60)
        print("🔮 AI PREDICTION EXAMPLES ON REAL DATA")
        print("=" * 60)
        
        # Select different types of products for demo
        test_cases = []
        
        # High performer - should not promote
        high_performers = df[df['should_promote'] == False]
        if len(high_performers) > 0:
            test_cases.append({
                'name': f"High Performer: {high_performers.iloc[0]['Libelle'][:40]}",
                'data': high_performers.iloc[0]
            })
        
        # Slow mover - should promote
        slow_movers = df[df['should_promote'] == True]
        if len(slow_movers) > 0:
            test_cases.append({
                'name': f"Needs Promotion: {slow_movers.iloc[0]['Libelle'][:40]}",
                'data': slow_movers.iloc[0]
            })
        
        # Medium performer (random sample)
        if len(df) > 2:
            random_product = df.sample(1).iloc[0]
            test_cases.append({
                'name': f"Random Product: {random_product['Libelle'][:40]}",
                'data': random_product
            })
        
        # Run predictions for each test case
        recommendations_df = pd.DataFrame()
        
        for i, test_case in enumerate(test_cases, 1):
            product_data = test_case['data']
            print(f"\n🔍 Test Case {i}: {test_case['name']}")
            print("-" * 50)
            
            # Current product stats
            print(f"Current Price: ${product_data['current_price']:.2f}")
            print(f"Current Stock: {int(product_data['current_stock'])} units")
            print(f"Sales (90d): {int(product_data['total_sales_90d'])} units")
            print(f"Rotation: {product_data['rotation']:.2f}")
            print(f"Stock Coverage: {product_data['stock_coverage_days']:.1f} days")
            print(f"Days Since Last Promo: {int(product_data['days_since_last_promo'])}")
            
            # Get AI prediction
            prediction = self.predict_promotion(product_data)
            
            # Display recommendation
            if prediction['should_promote']:
                print(f"\\n✅ RECOMMENDATION: PROMOTE THIS PRODUCT")
                print(f"   Suggested Discount: {prediction['optimal_discount']:.1%}")
                print(f"   Expected Volume Increase: +{prediction['predicted_sales_lift']:.0%}")
                print(f"   Expected Revenue Impact: +{prediction['predicted_sales_lift']:.0%}")
                print(f"   Confidence: {prediction['confidence_score']:.1%}")
            else:
                print(f"\\n❌ RECOMMENDATION: NO PROMOTION NEEDED")
                print(f"   Confidence: {prediction['confidence_score']:.1%}")
            
            print(f"   Reasoning: {prediction['recommendation_reason']}")
            
            # Add to recommendations DataFrame
            rec_data = {
                'product_name': test_case['name'],
                'code_article': product_data.get('CodeArticle', 'N/A'),
                'current_price': product_data['current_price'],
                'current_stock': product_data['current_stock'],
                'should_promote': prediction['should_promote'],
                'suggested_discount': prediction['optimal_discount'],
                'confidence': prediction['confidence_score'],
                'reasoning': prediction['recommendation_reason']
            }
            recommendations_df = pd.concat([recommendations_df, pd.DataFrame([rec_data])], ignore_index=True)
        
        # Show model performance
        if metrics:
            print("\n" + "=" * 60)
            print("🤖 AI MODEL PERFORMANCE")
            print("=" * 60)
            print(f"Promotion Classifier Accuracy: {metrics.get('promotion_accuracy', 0):.1%}")
            if 'discount_r2' in metrics:
                print(f"Discount Prediction R²: {metrics['discount_r2']:.3f}")
            if 'impact_r2' in metrics:
                print(f"Impact Prediction R²: {metrics['impact_r2']:.3f}")
        
        # Summary insights
        print("\n" + "=" * 60)
        print("📊 BUSINESS INSIGHTS FROM REAL DATA")
        print("=" * 60)
        
        promo_needed = df['should_promote'].sum()
        total_products = len(df)
        
        print(f"• Total Products Analyzed: {total_products}")
        print(f"• Products Needing Promotion: {promo_needed} ({promo_needed/total_products:.1%})")
        print(f"• Average Stock Coverage: {df['stock_coverage_days'].mean():.1f} days")
        print(f"• Products with High Stock (>30 days coverage): {len(df[df['stock_coverage_days'] > 30])}")
        print(f"• Products with Declining Sales: {len(df[df['sales_trend'] < -0.1])}")
        
        # Category-specific insights
        if 'CategorieName' in df.columns:
            print("\n📈 Category-Specific Promotion Needs:")
            category_promo = df.groupby('CategorieName')['should_promote'].agg(['count', 'sum', 'mean'])
            category_promo['promo_rate'] = category_promo['mean']
            category_promo = category_promo.sort_values('promo_rate', ascending=False).head(5)
            
            for cat, row in category_promo.iterrows():
                print(f"   {cat}: {int(row['sum'])}/{int(row['count'])} products ({row['promo_rate']:.1%})")
        
        print("\n💡 NEXT STEPS FOR REAL IMPLEMENTATION:")
        print("1. ✅ Real database integration completed")
        print("2. 🔄 Fine-tune model with more historical promotion results")
        print("3. 🚀 Deploy Flask API for production use")
        print("4. 📱 Integrate with Angular dashboard")
        print("5. ⏰ Set up automated daily/weekly analysis")
        print("6. 📊 Add real-time sales monitoring")
        
        return df, recommendations_df
def main():
    """Run the AI promotion demo with real database data"""
    print("🚀 Starting AI Promotion Demo with Real Database Integration")
    print("=" * 60)
    
    # Initialize demo with real database connection
    demo = AIPromotionDemo()
    
    # Run the demo with real data
    products_df, recommendations_df = demo.run_demo_with_real_data()
    
    # Optional: Save results
    try:
        products_df.to_csv('real_data_products.csv', index=False)
        recommendations_df.to_csv('real_data_recommendations.csv', index=False)
        print("\n💾 Results saved to real_data_products.csv and real_data_recommendations.csv")
    except Exception as e:
        print(f"\n⚠️  Could not save CSV files: {e}")

if __name__ == "__main__":
    main()
