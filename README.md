# Smart Promotion Model API

An intelligent Flask application that analyzes your SQL Server database and generates optimized promotions to maximize revenue and CA (Chiffre d'Affaires).

## Features

- **Database Analysis**: Automatically reads all database tables and analyzes structure
- **Smart Promotion Calculation**: Uses advanced algorithms to calculate optimal promotions
- **Category-Based Analysis**: Analyzes all products under a specific category
- **Revenue Optimization**: Generates promotions to maximize CA while maintaining profit margins
- **Intelligent Filtering**: Only creates promotions for products that actually need them
- **Database Integration**: Automatically saves promotions to the database

## Your Connection String

```
Server=DESKTOP-S22JEMV\SQLEXPRESS;Database=SmartPromoDb_v2024;Trusted_Connection=True;
```

## Promotion Algorithm

The system uses a sophisticated algorithm that considers:

1. **Sales Trends**: Analyzes if sales are increasing, decreasing, or stable
2. **Stock Levels**: Considers inventory levels for promotion urgency
3. **Profit Margins**: Ensures promotions maintain minimum profit thresholds
4. **Sales Performance**: Evaluates monthly sales averages
5. **Revenue Impact**: Projects revenue increase from promotions

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prerequisites

- SQL Server Express running on DESKTOP-S22JEMV\SQLEXPRESS
- Database SmartPromoDb_v2024 must exist
- Windows Authentication enabled
- ODBC Driver 17 for SQL Server installed

### 3. Run the Application

```bash
python promotion_model.py
```

## API Endpoints

### Core Endpoints

- `GET /` - API information and available endpoints
- `GET /database-structure` - Analyze complete database structure
- `GET /categories` - List all available categories

### Analysis Endpoints

- `GET /analyze-category/<category_id>` - Analyze products in category (no promotion generation)
- `POST /generate-promotions` - Generate and save promotions for category

### Promotion Management

- `GET /promotions` - List all active promotions

## Usage Examples

### 1. Analyze Category Products
```bash
GET /analyze-category/1
```
Returns detailed analysis of all products in category 1, including:
- Sales trends
- Stock levels
- Profit margins
- Promotion recommendations

### 2. Generate Promotions for Category
```bash
POST /generate-promotions
Content-Type: application/json

{
  "category_id": 1
}
```
Analyzes category 1 and generates optimized promotions for products that need them.

### 3. View Active Promotions
```bash
GET /promotions
```
Lists all currently active promotions with details.

## Promotion Logic

### When Products Need Promotion:
- Sales are decreasing (trend analysis)
- Sales are stable but low (< 10 units/month)
- High stock levels with low sales
- Very low sales (< 5 units/month)

### Discount Calculation:
- Base discount: 5%
- +10% for decreasing sales
- +8% for very low sales (< 5/month)
- +5% for low sales (< 15/month)
- +7% for high stock (> 100 units)
- +3% for medium-high stock (> 50 units)

### Constraints:
- Minimum profit margin: 10%
- Maximum discount: 30%
- Maintains profitability while maximizing sales

## Database Tables Used

The system automatically analyzes your database structure and uses:
- `Articles` - Product information
- `Categories` - Product categories
- `Ventes` - Sales data for trend analysis
- `Stocks` - Inventory levels
- `Promotions` - Generated promotions (auto-created)

## Response Format

All endpoints return JSON responses with:
- Status indicator
- Relevant data
- Error messages (when applicable)

## Troubleshooting

1. **Database Connection Issues**:
   - Verify SQL Server is running
   - Check Windows Authentication
   - Confirm database exists

2. **No Promotions Generated**:
   - Products may not need promotions
   - Check category has products
   - Verify sales data exists

3. **Permission Issues**:
   - Ensure user has database write permissions
   - Check table creation permissions

## Security Features

- Windows Authentication (no credentials stored)
- Input validation and sanitization
- Error handling and logging
- SQL injection protection via parameterized queries
