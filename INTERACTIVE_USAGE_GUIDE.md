# Interactive AI Promotion Model - Usage Guide

## New Features Added

### 1. Category Selection

- The model now loads all data from the database first
- Displays available product categories with statistics
- Allows you to select which category to create promotions for
- Shows category performance metrics (product count, promotion needs, stock levels)

### 2. Date Prediction

- Input: Start date for the promotion
- Output: AI-predicted optimal end date
- Considers category characteristics, stock levels, and discount rates
- Provides reasoning for the predicted duration

### 3. Interactive Mode

- User-friendly interface for category and date selection
- Real-time analysis and recommendations
- Detailed promotion planning with ROI calculations

## How to Use

### Running the Interactive Demo

```bash
python ai_demo.py
```

When prompted:

1. Choose demo mode (1 for interactive, 2 for full analysis)
2. Select a category number from the list
3. Enter promotion start date (YYYY-MM-DD format) or press Enter for today

### Example Workflow

1. **Load Data**: Model connects to SmartPromoDb_v2024 and loads all products
2. **View Categories**: See all available categories with stats:
   ```
   1. Dresses          (45 products, 12 need promotion, 67d stock)
   2. T-Shirts         (38 products, 8 need promotion, 23d stock)
   3. Jeans            (29 products, 5 need promotion, 45d stock)
   ```
3. **Select Category**: Choose number (e.g., "1" for Dresses)
4. **Set Start Date**: Enter "2025-07-15" or press Enter for today
5. **Get Recommendations**: Model analyzes and provides:
   - Products needing promotion in that category
   - Optimal discount rates
   - Predicted sales lift
   - Expected revenue impact
   - **Predicted end date** (e.g., "2025-07-29")
   - Duration reasoning

### Output Example

```
🎯 ANALYZING CATEGORY: Dresses
📅 Promotion Start Date: 2025-07-15

📊 Found 45 products in Dresses

📈 Category Performance Overview:
   Average Price: $156.78
   Total Stock: 1,234 units
   Products Needing Promotion: 12/45 (27%)

🎯 PROMOTION RECOMMENDATIONS FOR 12 PRODUCTS:

 1. Summer Floral Dress
    Price: $89.99 | Stock: 45 | Coverage: 120d
    💰 Suggested Discount: 20%
    📈 Expected Sales Lift: +45%
    💵 Revenue Impact: +$234.50/month
    🎯 Confidence: 87%

📋 CATEGORY PROMOTION SUMMARY
Category: Dresses
Start Date: 2025-07-15
Predicted End Date: 2025-07-29
Duration: 14 days
Reasoning: Duration extended to 14 days due to: standard promotion duration
```

## Key Features

### Date Prediction Algorithm

The model considers:

- **Discount Rate**: Higher discounts → longer promotions
- **Stock Coverage**: Excess inventory → extended duration
- **Product Rotation**: Slow movers → longer exposure needed
- **Price Point**: Premium products → shorter promotions
- **Category Characteristics**: Average performance metrics

### Duration Bounds

- Minimum: 7 days
- Maximum: 28 days
- Default: 14 days

### Business Logic

- High stock (>90d coverage) → +10 days
- High discount (>25%) → +7 days
- Low rotation (<0.2) → +7 days
- Premium pricing (>$200) → -5 days

## Files Created/Modified

1. **ai_demo.py** - Main model with new interactive features
2. **interactive_demo_example.py** - Example usage script
3. **INTERACTIVE_USAGE_GUIDE.md** - This guide

## Technical Implementation

### New Methods Added:

- `get_available_categories()` - Extract categories from data
- `predict_promotion_end_date()` - AI-powered date prediction
- `run_interactive_promotion_demo()` - Interactive user interface
- `analyze_category_promotion()` - Category-specific analysis

### Enhanced Features:

- User input validation
- Error handling for date formats
- Graceful fallbacks for missing data
- Comprehensive promotion planning

## Next Steps

1. **Test the Interactive Mode**: Run `python ai_demo.py` and try different categories
2. **Validate Date Predictions**: Check if predicted durations make business sense
3. **Integrate with Frontend**: Use this API structure for Angular integration
4. **Add More Business Rules**: Customize duration logic for your specific needs

## API Structure for Integration

The interactive demo returns structured data that can be easily integrated:

```python
recommendations_df = {
    'product_name': str,
    'code_article': str,
    'current_price': float,
    'suggested_discount': float,
    'predicted_sales_lift': float,
    'start_date': str,
    'end_date': str,        # AI-predicted
    'duration_days': int,    # AI-calculated
    'category': str,
    'expected_revenue_increase': float,
    'confidence': float
}
```

This structure is ready for database insertion and frontend display.
