# NYC Bicycle Traffic Modeling 

Predicting NYC bridge cyclist traffic using regression, weather data, and classification.

## Overview
This project uses real-world 2016 NYC bicycle traffic data across four bridges to:
- Predict total cyclists from individual bridge data (Linear Regression)
- Assess predictive power of weather features (Regression on Temp & Precip)
- Classify day of the week using Random Forest (Bridge usage patterns)

## Results
- Max R² = **0.951** (Williamsburg Bridge)
- Weather-based R² = **0.575**
- Day-of-week prediction accuracy = **23.3%**

## Hypothesis Testing
Used Z-test to verify if average R² across bridges > 0.25  
→ Null hypothesis rejected (**p-value ≈ 0.0000**)

## Tools Used
- Python, pandas, NumPy
- scikit-learn (Linear Regression, Random Forest)
- matplotlib
- scipy.stats
