from scipy.stats import poisson, chi2
import numpy as np
import matplotlib.pyplot as plt

def solve():
    # Given observed data
    observed_freq = [1, 4, 13, 19, 16, 15, 9, 12, 7, 2, 1, 1]
    n = 100  # Total number of observations

    # Expected frequencies under the Poisson distribution assumption
    mu = 5.59
    expected_freq = [poisson.pmf(i + 1, mu) * n for i in range(len(observed_freq))]

    # Categories
    categories = np.arange(1, len(observed_freq) + 1)

    # Create bar plot for observed and expected frequencies
    plt.figure(figsize=(8, 6))

    plt.bar(categories - 0.2, observed_freq, width=0.4, align='center', label='Observed', color='blue')
    plt.bar(categories + 0.2, expected_freq, width=0.4, align='center', label='Expected', color='orange')

    plt.xlabel('Categories')
    plt.ylabel('Frequencies')
    plt.title('Observed vs Expected Frequencies')
    plt.legend()
    plt.xticks(categories)
    plt.grid(axis='y')

    # Show the plot
    plt.show()

    # Calculating the chi-square statistic
    chi_square_stat = sum((np.array(observed_freq) - np.array(expected_freq))**2 / np.array(expected_freq))

    # Degrees of freedom: number of categories - 1 - number of parameters estimated from data
    degrees_of_freedom = len(observed_freq) - 1 - 1  # For Poisson distribution, it has 1 parameter (mean)

    # Critical value at alpha = 0.05 significance level
    alpha = 0.05
    critical_value = chi2.ppf(1 - alpha, degrees_of_freedom)

    # Compare chi-square statistic with critical value
    print(f"Chi-square statistic: {chi_square_stat}")
    print(f"Critical value at alpha = 0.05 and df = {degrees_of_freedom}: {critical_value}")

    if chi_square_stat < critical_value:
        print("The observed frequencies fit a Poisson distribution (fail to reject H0)")
    else:
        print("The observed frequencies do not fit a Poisson distribution (reject H0)")


