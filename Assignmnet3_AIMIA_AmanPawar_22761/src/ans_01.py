import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def solve(lesion_volume_A, lesion_volume_B, alpha):

    # Calculate the differences between the paired measurements
    differences = lesion_volume_A - lesion_volume_B

    # # Perform paired sample t-test
    # t_statistic, p_value = stats.ttest_rel(lesion_volume_A, lesion_volume_B)

    # Assuming 'differences' represents the differences between paired measurements
    t_statistic, p_value = stats.ttest_rel(lesion_volume_A, lesion_volume_B, alternative='greater')


    # Define the significance level (alpha)
    # alpha = 0.05

    # Calculate critical value
    critical_value = stats.t.ppf(1 - alpha/2, df=len(differences)-1)

    # Define x values for plotting
    x_values = np.linspace(-5, 5, 1000)

    # Plotting the critical region
    plt.figure(figsize=(8, 5))
    plt.hist(differences, bins=5, density=True, alpha=0.6, color='skyblue', label="diff in lesion vol")


    # Highlight the critical region between t_statistic and the critical value
    plt.fill_between(x_values[(x_values > t_statistic) & (x_values < critical_value)], 0, 
                    stats.norm.pdf(x_values[(x_values > t_statistic) & (x_values < critical_value)]), color='red', alpha=0.3, label="Critical Region")


    plt.title('Paired t-test - Critical Region')
    plt.xlabel('Difference in Lesion Volumes (Method A - Method B)')
    plt.ylabel('Density')
    plt.axvline(x=critical_value, color='black', linestyle='--', label=f'Critical Value: {critical_value:.2f}')
    plt.axvline(x=-critical_value, color='black', linestyle='--')
    plt.legend()
    plt.show()

    print(f"Test Statistic: {t_statistic:.4f}")
    print(f"Critical Value: ±{critical_value:.4f}")
    print(f"P-value: {p_value:.4f}")


def solve2(lesion_volume_A, lesion_volume_B, alpha):
    # Calculate the differences between the paired measurements
    differences = lesion_volume_A - lesion_volume_B

    # # Perform paired sample t-test
    # t_statistic, p_value = stats.ttest_rel(lesion_volume_A, lesion_volume_B)

    # Assuming 'differences' represents the differences between paired measurements
    t_statistic, p_value = stats.ttest_rel(lesion_volume_A, lesion_volume_B, alternative='greater')


    # Define the significance level (alpha)
    # alpha = 0.05

    # Calculate critical value
    critical_value = stats.t.ppf(1 - alpha, df=len(differences)-1)

    # Define x values for plotting
    x_values = np.linspace(-5, 5, 1000)

    # Plotting the critical region
    plt.figure(figsize=(8, 5))
    plt.hist(differences, bins=5, density=True, alpha=0.6, color='skyblue', label="diff in lesion vol")


    # Highlight the critical region between t_statistic and the critical value
    plt.fill_between(x_values[(x_values > t_statistic) & (x_values < critical_value)], 0, 
                    stats.norm.pdf(x_values[(x_values > t_statistic) & (x_values < critical_value)]), color='red', alpha=0.3, label="Critical Region")


    plt.title('Paired t-test - Critical Region')
    plt.xlabel('Difference in Lesion Volumes (Method A - Method B)')
    plt.ylabel('Density')
    plt.axvline(x=critical_value, color='black', linestyle='--', label=f'Critical Value: {critical_value:.2f}')
    plt.axvline(x=-critical_value, color='black', linestyle='--')
    plt.legend()
    plt.show()

    print(f"Test Statistic: {t_statistic:.4f}")
    print(f"Critical Value: ±{critical_value:.4f}")
    print(f"P-value: {p_value:.4f}")