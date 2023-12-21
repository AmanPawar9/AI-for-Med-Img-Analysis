import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def solve():
    # Define the data
    data = {
        'BMI': [
            22.56, 16.54, 18.58, 18.20, 25.02, 24.64, 15.44, 14.56, 23.66, 24.62, 16.12, 15.54, 17.22, 19.06, 16.88, 16.82, 22.58, 20.12, 17.58, 14.56,
            25.64, 22.50, 17.82, 19.74, 28.84, 24.48, 15.76, 17.48, 26.00, 25.52, 12.96, 16.46, 26.02, 24.76, 15.00, 16.44, 23.24, 20.62, 19.54, 15.70
        ],
        'Nutrient': ['DE'] * 20 + ['DR'] * 20,
        'Age': ['T','A','M','O'] * 10 
    }

    df = pd.DataFrame(data)

    # Fit the ANOVA model
    formula = 'BMI ~ C(Nutrient) + C(Age) + C(Nutrient):C(Age)'
    model = ols(formula, df).fit()
    anova_results = anova_lm(model)

    # Display the ANOVA table
    print(anova_results)

    # Extract F-statistics and p-values
    f_nutrient = anova_results['F']['C(Nutrient)']
    f_age = anova_results['F']['C(Age)']
    f_interaction = anova_results['F']['C(Nutrient):C(Age)']

    p_nutrient = anova_results['PR(>F)']['C(Nutrient)']
    p_age = anova_results['PR(>F)']['C(Age)']
    p_interaction = anova_results['PR(>F)']['C(Nutrient):C(Age)']

    # Print results
    print(f"\nF-statistic for Nutrient: {f_nutrient}, p-value: {p_nutrient}")
    print(f"F-statistic for Age: {f_age}, p-value: {p_age}")
    print(f"F-statistic for Interaction: {f_interaction}, p-value: {p_interaction}")

