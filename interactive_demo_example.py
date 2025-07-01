"""
Interactive AI Promotion Demo Example
=====================================

This script demonstrates how to use the new interactive features:
1. Select a product category
2. Choose promotion start date
3. Get AI-predicted end date and recommendations

Run: python interactive_demo_example.py
"""

from ai_demo import AIPromotionDemo
import pandas as pd
from datetime import datetime, timedelta


def demo_category_selection():
    """Demonstrate the category selection and date prediction features"""
    print("🎯 INTERACTIVE AI PROMOTION DEMO EXAMPLE")
    print("=" * 50)

    # Initialize the AI demo
    demo = AIPromotionDemo()

    # Create sample data for demonstration
    sample_data = demo.generate_sample_data(100)
    print(f"📊 Generated {len(sample_data)} sample products for demo")

    # Train the models
    demo.train_models(sample_data)

    # Get available categories
    categories = demo.get_available_categories(sample_data)
    print(f"\n📋 Available Categories: {categories}")

    # Demonstrate category analysis
    selected_category = categories[0]  # Pick first category
    start_date = "2025-07-15"  # Example start date

    print(f"\n🔍 Analyzing category: {selected_category}")
    print(f"📅 Start date: {start_date}")

    # Run category analysis
    category_data, recommendations = demo.analyze_category_promotion(
        sample_data, selected_category, start_date
    )

    if not recommendations.empty:
        print(f"\n✅ Generated {len(recommendations)} recommendations")
        print(f"📅 Predicted end date: {recommendations['end_date'].iloc[0]}")
        print(f"⏱️  Duration: {recommendations['duration_days'].iloc[0]} days")

        # Show top 3 recommendations
        top_recommendations = recommendations.head(3)
        print(f"\n🏆 Top 3 Recommendations:")
        for idx, row in top_recommendations.iterrows():
            print(f"   {idx+1}. {row['product_name'][:30]}")
            print(f"      Discount: {row['suggested_discount']:.1%}")
            print(f"      Expected lift: +{row['predicted_sales_lift']:.0%}")
            print(f"      Revenue impact: +${row['expected_revenue_increase']:.2f}")

    return category_data, recommendations


def demo_date_prediction():
    """Demonstrate the date prediction functionality"""
    print("\n" + "=" * 50)
    print("📅 DATE PREDICTION DEMO")
    print("=" * 50)

    demo = AIPromotionDemo()

    # Create sample category data
    sample_data = pd.DataFrame(
        {
            "rotation": [0.2, 0.3, 0.1, 0.4],
            "stock_coverage_days": [120, 45, 180, 30],
            "current_price": [150, 80, 250, 120],
        }
    )

    # Test different scenarios
    scenarios = [
        {"start_date": "2025-07-01", "discount": 0.15, "scenario": "Normal promotion"},
        {
            "start_date": "2025-08-01",
            "discount": 0.25,
            "scenario": "High discount clearance",
        },
        {
            "start_date": "2025-09-01",
            "discount": 0.10,
            "scenario": "Premium product promotion",
        },
    ]

    for scenario in scenarios:
        result = demo.predict_promotion_end_date(
            scenario["start_date"], sample_data, scenario["discount"]
        )

        print(f"\n📋 {scenario['scenario']}:")
        print(f"   Start: {scenario['start_date']}")
        print(f"   End: {result['end_date']}")
        print(f"   Duration: {result['duration_days']} days")
        print(f"   Reasoning: {result['reasoning']}")


if __name__ == "__main__":
    # Run the demos
    category_data, recommendations = demo_category_selection()
    demo_date_prediction()

    print(f"\n🎉 Demo completed successfully!")
    print(f"💡 To run the full interactive demo, use: python ai_demo.py")
