# Utility functions
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.stats.power as smp
import matplotlib.pyplot as plt

def make_images(dataloader):
    print("Visualtizing Data from the DataLoader...")
    data_iterator = iter(dataloader)
    batch = next(data_iterator)
    images, labels = batch

    images = images.cpu().numpy()
    labels = labels.numpy()
    print(f"The Shape of Image is : {images[0].shape}")
    # Display the first 5 images in a horizontal layout
    plt.figure(figsize=(15, 7))
    # Plot images
    for i in range(5):
        plt.subplot(2, 5, i + 1)  # First row for images
        image = (images[i] +1)/2.0
        label = labels[i]
        image = np.moveaxis(image, 0, -1)  # Transpose to HWC format
        plt.imshow(image)
        plt.title(f"Label:{label[0]}")
        plt.axis('off')

    plt.show()


def plot_loss_curve(file_path):
    train_losses = []
    val_losses = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Skip the first line (header)
        for line in lines[1:]:
            train_loss, val_loss = line.strip().split('\t')
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses,'--r', label='Train Loss')
    plt.plot(epochs, val_losses, '--b',label='Validation Loss')
    plt.title('Train and Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def solve(lesion_volume_A, lesion_volume_B, alpha):

    # Calculate the differences between the paired measurements
    differences = lesion_volume_A - lesion_volume_B

    # # Perform paired sample t-test
    # t_statistic, p_value = stats.ttest_rel(lesion_volume_A, lesion_volume_B)

    # Assuming 'differences' represents the differences between paired measurements
    t_statistic, p_value = stats.ttest_rel(lesion_volume_A, lesion_volume_B, alternative='greater')


    # Calculate critical value
    critical_value = stats.t.ppf(1 - alpha/2, df=len(differences)-1)

    # Define x values for plotting
    x_values = np.linspace(-5, 5, 1000)

    # Plotting the critical region
    plt.figure(figsize=(8, 5))
    plt.hist(differences, bins=5, density=True, alpha=0.6, color='skyblue', label="Diff in Images")


    # Highlight the critical region between t_statistic and the critical value
    plt.fill_between(x_values[(x_values > t_statistic) & (x_values < critical_value)], 0, 
                    stats.norm.pdf(x_values[(x_values > t_statistic) & (x_values < critical_value)]), color='red', alpha=0.3, label="Critical Region")


    plt.title('Paired t-test - Critical Region')
    plt.xlabel('Difference in Images (Method A - Method B)')
    plt.ylabel('Density')
    plt.axvline(x=critical_value, color='black', linestyle='--', label=f'Critical Value: {critical_value:.2f}')
    plt.axvline(x=-critical_value, color='black', linestyle='--')
    plt.legend()
    plt.show()

    print(f"Test Statistic: {t_statistic:.4f}")
    print(f"Critical Value: ±{critical_value:.4f}")
    print(f"P-value: {p_value:.4f}")


def power_analysis_1b():
    # Define parameters for the power analysis
    effect_sizes = [0.2, 0.42, 0.8]  # Effect sizes to test
    sample_sizes = [32,64,512,1024,3421]    # Sample sizes to test
    alpha = 0.05                    # Significance level

    # Function to perform power analysis for t-tests
    def perform_power_analysis(effect_sizes, sample_sizes):
        results = []
        for effect_size in effect_sizes:
            for sample_size in sample_sizes:
                # Calculate power for real vs. generated images comparison (Gen1)
                power_real_gen1 = smp.TTestIndPower().solve_power(effect_size=effect_size, nobs1=sample_size, alpha=alpha)

                results.append({
                    'Effect Size': effect_size,
                    'Sample Size': sample_size,
                    'Power': power_real_gen1,

                })

        # Create DataFrame from the results list
        df = pd.DataFrame(results)

        return df

    # Perform power analysis and display results in a table
    power_analysis_df = perform_power_analysis(effect_sizes, sample_sizes)
    return power_analysis_df