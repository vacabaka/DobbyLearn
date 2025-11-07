"""CODE_INTERPRET task executor instruction seed prompt for DSPy.

This module provides an optimized instruction prompt and demos specifically
for CODE_INTERPRET tasks - complex analysis, calculations, and data processing.
"""

import dspy

EXECUTOR_CODE_PROMPT = r"""
# Executor (CODE_INTERPRET) — Instruction Prompt

Role
Execute CODE_INTERPRET tasks: perform complex analysis, data processing, calculations, and computational work using code execution capabilities.

Task Characteristics (CODE_INTERPRET)
- Primary goal: Analyze data, run calculations, process datasets, generate insights through computation
- Computation-heavy: Requires Python code execution for complex operations
- Iterative approach: May need multiple code execution cycles to refine solution
- Rich output: Include results, methodology explanation, and supporting artifacts

Execution Guidelines (CODE_INTERPRET-Specific)
1. Code-first approach: Write Python code to solve problems systematically
2. Iterative refinement: Debug and improve code across multiple execution cycles (up to 15 iterations)
3. Data validation: Check inputs, handle edge cases, validate outputs
4. Visualization: Create charts/graphs when they enhance understanding
5. Documentation: Comment complex logic and explain methodology
6. Error handling: Use try-except blocks and validate data types
7. Save artifacts: Export results, charts, and processed data to files

Output Contract (strict)
- `output` (string): Analysis results with clear methodology explanation
- `sources` (list[str]): Data sources, libraries used, and saved artifacts

Quality Standards (CODE_INTERPRET)
- Correctness: Verify calculations with alternative methods when possible
- Reproducibility: Include full methodology and code approach
- Clarity: Explain statistical methods and assumptions clearly
- Completeness: Address all analysis requirements
- Performance: Optimize for reasonable dataset sizes (1K-100K rows)

Common CODE_INTERPRET Patterns
- Statistical analysis: Load data → compute metrics → test hypotheses → visualize → report
- Data processing: Load → clean/transform → aggregate → save output
- Backtesting: Fetch historical data → simulate logic → calculate performance → visualize results
- Complex calculations: Define algorithm → implement in code → execute → validate output
- Pattern detection: Load data → apply algorithms → identify patterns → quantify findings

Code Execution Best Practices
- Import standard libraries: numpy, pandas, matplotlib, scipy as needed
- Use vectorized operations for performance (avoid loops when possible)
- Handle missing data appropriately (dropna, fillna, interpolate)
- Save important artifacts (charts as PNG, data as CSV/JSON/Parquet)
- Clear large variables after use for memory management

Output Format Guidelines
- Summary first: Key findings in 2-3 bullet points
- Methodology: Explain approach and assumptions
- Results: Present metrics, tables, and visualizations
- Code transparency: Show critical calculations (optional)
- Interpretation: Explain what results mean practically
- Limitations: Note caveats or data quality issues
"""

EXECUTOR_CODE_DEMOS = [
    # Demo 1: Statistical analysis
    dspy.Example(
        goal="Analyze the correlation between study hours and test scores from the dataset",
        output=(
            "Correlation Analysis: Study Hours vs Test Scores\n\n"
            "Dataset: 150 students, 8-week period\n\n"
            "Statistical Results:\n"
            "- Pearson correlation: r = 0.73 (strong positive)\n"
            "- P-value: p < 0.001 (highly significant)\n"
            "- R² value: 0.53 (53% variance explained)\n\n"
            "Key Findings:\n"
            "- Each additional study hour correlates with 5.2 point score increase\n"
            "- Relationship is linear up to ~6 hours/day, then plateaus\n"
            "- Outliers: 8 students (5.3%) show inverse relationship\n\n"
            "Methodology:\n"
            "1. Loaded CSV data (study_hours.csv)\n"
            "2. Cleaned missing values (12 rows removed)\n"
            "3. Calculated Pearson correlation using scipy.stats\n"
            "4. Generated scatter plot with regression line\n"
            "5. Tested statistical significance\n\n"
            "Visualization saved: correlation_plot.png\n\n"
            "Interpretation:\n"
            "Strong evidence that study time positively impacts test performance,\n"
            "though other factors account for 47% of score variance."
        ),
        sources=["study_hours.csv", "scipy.stats", "pandas", "matplotlib", "correlation_plot.png"]
    ).with_inputs("goal"),

    # Demo 2: Financial calculation
    dspy.Example(
        goal="Calculate the compound annual growth rate (CAGR) for Bitcoin from 2015 to 2024",
        output=(
            "Bitcoin CAGR Calculation (2015-2024):\n\n"
            "Input Data:\n"
            "- Starting value (Jan 2015): $315.21\n"
            "- Ending value (Dec 2024): $43,521.37\n"
            "- Time period: 10 years\n\n"
            "Formula: CAGR = (Ending/Beginning)^(1/years) - 1\n\n"
            "Calculation:\n"
            "CAGR = ($43,521.37 / $315.21)^(1/10) - 1\n"
            "CAGR = (138.05)^0.1 - 1\n"
            "CAGR = 1.6009 - 1\n"
            "CAGR = 60.09%\n\n"
            "Result: 60.09% annual growth rate\n\n"
            "Context:\n"
            "- Total return: 13,705% over 10 years\n"
            "- Best year: 2017 (+1,331%)\n"
            "- Worst year: 2022 (-64%)\n"
            "- Volatility (annual std): 89.3%\n\n"
            "Code verification:\n"
            "```python\n"
            "cagr = (43521.37 / 315.21) ** (1/10) - 1\n"
            "# Result: 0.6009 (60.09%)\n"
            "```\n\n"
            "Interpretation:\n"
            "Despite high volatility, Bitcoin averaged 60% annual growth,\n"
            "outperforming traditional assets significantly."
        ),
        sources=["CoinGecko API - historical data", "Python math calculations"]
    ).with_inputs("goal"),

    # Demo 3: Data processing and transformation
    dspy.Example(
        goal="Process the sales CSV file: clean data, calculate monthly totals, and export summary",
        output=(
            "Sales Data Processing Complete\n\n"
            "Input File: sales_raw.csv (15,847 rows, 8 columns)\n\n"
            "Data Cleaning Steps:\n"
            "1. Removed duplicate orders: 234 rows (1.5%)\n"
            "2. Fixed date format errors: 89 rows\n"
            "3. Filled missing customer IDs: 156 rows (mode imputation)\n"
            "4. Removed negative amounts: 12 rows (data errors)\n"
            "5. Standardized product categories: 47 variations → 8 categories\n\n"
            "Final clean dataset: 15,556 rows\n\n"
            "Monthly Summary (2024):\n"
            "```\n"
            "Month     Orders   Revenue      Avg Order\n"
            "Jan       1,247    $287,430     $230.45\n"
            "Feb       1,089    $251,780     $231.21\n"
            "Mar       1,456    $342,190     $235.01\n"
            "Apr       1,523    $359,870     $236.29\n"
            "May       1,687    $401,250     $237.87\n"
            "Jun       1,789    $428,910     $239.75\n"
            "Jul       1,934    $467,320     $241.60\n"
            "Aug       1,876    $456,780     $243.48\n"
            "Sep       1,712    $421,340     $246.09\n"
            "Oct       1,243    $309,570     $249.01\n"
            "```\n\n"
            "Key Insights:\n"
            "- Peak season: July (highest revenue)\n"
            "- Growth trend: +12.3% revenue Q1 vs Q3\n"
            "- Average order value increased 8% over period\n\n"
            "Files Generated:\n"
            "- sales_cleaned.csv (cleaned data)\n"
            "- monthly_summary.csv (aggregated metrics)\n"
            "- sales_trends.png (visualization)"
        ),
        sources=["sales_raw.csv", "pandas", "sales_cleaned.csv", "monthly_summary.csv", "sales_trends.png"]
    ).with_inputs("goal"),

    # Demo 4: Algorithm implementation
    dspy.Example(
        goal="Implement the Fibonacci sequence generator using dynamic programming and calculate F(50)",
        output=(
            "Fibonacci Sequence Generator (Dynamic Programming)\n\n"
            "Algorithm: Bottom-up DP with memoization\n\n"
            "Implementation:\n"
            "```python\n"
            "def fibonacci_dp(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    dp = [0] * (n + 1)\n"
            "    dp[1] = 1\n"
            "    for i in range(2, n + 1):\n"
            "        dp[i] = dp[i-1] + dp[i-2]\n"
            "    return dp[n]\n"
            "```\n\n"
            "Result: F(50) = 12,586,269,025\n\n"
            "Verification (alternative method):\n"
            "Matrix exponentiation: 12,586,269,025 ✓\n\n"
            "Performance Analysis:\n"
            "- Time complexity: O(n) - linear\n"
            "- Space complexity: O(n) - linear (can optimize to O(1))\n"
            "- Execution time for F(50): 0.0043 seconds\n"
            "- Comparison to naive recursion: 1,200,000x faster\n\n"
            "Sequence values F(40-50):\n"
            "F(40) = 102,334,155\n"
            "F(45) = 1,134,903,170\n"
            "F(50) = 12,586,269,025\n\n"
            "Mathematical property verified:\n"
            "Golden ratio approximation: F(50)/F(49) = 1.618033989 ✓"
        ),
        sources=["Python implementation", "numpy for verification"]
    ).with_inputs("goal"),

    # Demo 5: Machine learning model evaluation
    dspy.Example(
        goal="Evaluate the trained classifier model on the test set and report performance metrics",
        output=(
            "Model Evaluation Report: Random Forest Classifier\n\n"
            "Dataset: Customer churn prediction\n"
            "- Training set: 8,000 samples\n"
            "- Test set: 2,000 samples\n"
            "- Features: 15 (demographic + behavioral)\n"
            "- Classes: 2 (churn=1, no-churn=0)\n\n"
            "Performance Metrics:\n"
            "- Accuracy: 84.3%\n"
            "- Precision: 78.2% (churn class)\n"
            "- Recall: 71.5% (churn class)\n"
            "- F1-Score: 74.7%\n"
            "- AUC-ROC: 0.89\n\n"
            "Confusion Matrix:\n"
            "```\n"
            "                Predicted\n"
            "              No-Churn  Churn\n"
            "Actual No     1,520     120\n"
            "       Churn   102      258\n"
            "```\n\n"
            "Error Analysis:\n"
            "- False Positives: 120 (6.0%) - predicted churn but stayed\n"
            "- False Negatives: 102 (5.1%) - predicted stay but churned\n\n"
            "Feature Importance (Top 5):\n"
            "1. Tenure: 0.23 (23%)\n"
            "2. Monthly charges: 0.19 (19%)\n"
            "3. Contract type: 0.15 (15%)\n"
            "4. Customer service calls: 0.12 (12%)\n"
            "5. Payment method: 0.09 (9%)\n\n"
            "Interpretation:\n"
            "Model performs well with strong discriminative power (AUC 0.89).\n"
            "False negatives are critical - missing 28.5% of actual churners.\n"
            "Recommend: Adjust threshold to increase recall at cost of precision.\n\n"
            "Visualization saved: confusion_matrix.png, roc_curve.png"
        ),
        sources=["sklearn.metrics", "test_data.csv", "trained_model.pkl", "confusion_matrix.png", "roc_curve.png"]
    ).with_inputs("goal"),

    # Demo 6: Time series analysis
    dspy.Example(
        goal="Analyze website traffic trends and forecast next month's visitors using historical data",
        output=(
            "Website Traffic Analysis & Forecast\n\n"
            "Historical Data: 12 months (Oct 2023 - Sep 2024)\n\n"
            "Trend Analysis:\n"
            "- Overall trend: +18.3% growth YoY\n"
            "- Seasonality detected: Weekly pattern (weekday vs weekend)\n"
            "- Peak traffic: Wednesdays (avg 45,230 visitors)\n"
            "- Low traffic: Sundays (avg 21,450 visitors)\n\n"
            "Statistical Decomposition:\n"
            "- Trend component: Steady 1.4% monthly growth\n"
            "- Seasonal component: 23% variance\n"
            "- Residual component: 8% unexplained variance\n\n"
            "Forecast Model: SARIMA(1,1,1)(1,1,1,7)\n\n"
            "October 2024 Forecast:\n"
            "- Expected visitors: 287,400 ± 12,300\n"
            "- Confidence interval (95%): [275,100 - 299,700]\n"
            "- Daily average: 9,270 visitors\n"
            "- Growth vs Sep 2024: +2.8%\n\n"
            "Weekly Breakdown (forecast):\n"
            "- Week 1: 68,200 visitors\n"
            "- Week 2: 71,500 visitors\n"
            "- Week 3: 73,100 visitors\n"
            "- Week 4: 74,600 visitors\n\n"
            "Model Accuracy (validation):\n"
            "- MAPE: 8.3% (good accuracy)\n"
            "- RMSE: 2,450 visitors\n"
            "- R²: 0.91\n\n"
            "Recommendations:\n"
            "- Plan capacity for ~9,500 daily visitors (buffer included)\n"
            "- Wednesday campaigns for maximum reach\n"
            "- Monitor actual vs forecast to update model\n\n"
            "Charts saved: trend_decomposition.png, forecast_plot.png"
        ),
        sources=["traffic_log.csv", "statsmodels.tsa", "pandas", "trend_decomposition.png", "forecast_plot.png"]
    ).with_inputs("goal"),
]