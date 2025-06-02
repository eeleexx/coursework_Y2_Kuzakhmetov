import json
import os
import numpy as np
from scipy import stats
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_lstm_results(ticker):
    """Load LSTM results for a specific ticker."""
    base_path = Path('results/lstm_analysis')
    results = {}

    # Load the main results file
    results_file = base_path / 'lstm_results.json'
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None

    with open(results_file, 'r') as f:
        all_results = json.load(f)

    if ticker not in all_results:
        print(f"No results found for {ticker}")
        return None

    # Debug: Print the structure of the data
    print(f"\nData structure for {ticker}:")
    for variation, data in all_results[ticker].items():
        print(f"\nVariation: {variation}")
        print("Keys in data:", data.keys())
        if 'metrics' in data:
            print("Keys in metrics:", data['metrics'].keys())
            if 'test_predictions' in data['metrics']:
                print(
                    f"Number of predictions: {len(data['metrics']['test_predictions'])}")
            if 'test_actual' in data['metrics']:
                print(
                    f"Number of actual values: {len(data['metrics']['test_actual'])}")

    return all_results[ticker]


def calculate_errors(predictions, actual):
    """Calculate absolute prediction errors."""
    # Convert nested predictions to flat array
    predictions_flat = np.array(
        [p[0] if isinstance(p, list) else p for p in predictions])
    actual_flat = np.array(actual)
    return np.abs(predictions_flat - actual_flat)


def perform_statistical_tests(ticker_results):
    """Perform Kruskal-Wallis and KS tests on prediction errors."""
    # Group variations by their prediction lengths
    length_groups = {}
    error_distributions = {}

    # Calculate errors for each feature combination
    for variation, data in ticker_results.items():
        if 'metrics' in data and 'test_predictions' in data['metrics'] and 'test_actual' in data['metrics']:
            errors = calculate_errors(
                data['metrics']['test_predictions'], data['metrics']['test_actual'])
            error_distributions[variation] = errors
            length = len(errors)
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append(variation)
            print(f"\nProcessed {variation}:")
            print(f"Number of errors: {len(errors)}")
            print(f"Mean error: {np.mean(errors):.4f}")

    if not error_distributions:
        print("No valid error distributions found!")
        return None

    results = {
        'group_tests': {},
        'error_means': {var: float(np.mean(errors)) for var, errors in error_distributions.items()},
        'error_stds': {var: float(np.std(errors)) for var, errors in error_distributions.items()}
    }

    # Perform tests within each length group
    for length, variations in length_groups.items():
        if len(variations) >= 2:
            groups = [error_distributions[var] for var in variations]
            h_stat, p_value_kw = stats.kruskal(*groups)

            # Convert numpy values to Python scalars
            h_stat = float(h_stat.item() if hasattr(
                h_stat, 'item') else h_stat)
            p_value_kw = float(p_value_kw.item() if hasattr(
                p_value_kw, 'item') else p_value_kw)

            results['group_tests'][f'length_{length}'] = {
                'variations': variations,
                'kruskal_wallis': {
                    'h_statistic': h_stat,
                    'p_value': p_value_kw
                },
                'ks_tests': {}
            }

            # Perform pairwise KS tests within the group
            for i, var1 in enumerate(variations):
                for var2 in variations[i+1:]:
                    ks_stat, p_value_ks = stats.ks_2samp(
                        error_distributions[var1],
                        error_distributions[var2]
                    )
                    # Convert numpy values to Python scalars
                    ks_stat = float(ks_stat.item() if hasattr(
                        ks_stat, 'item') else ks_stat)
                    p_value_ks = float(p_value_ks.item() if hasattr(
                        p_value_ks, 'item') else p_value_ks)

                    results['group_tests'][f'length_{length}']['ks_tests'][f"{var1}_vs_{var2}"] = {
                        'ks_statistic': ks_stat,
                        'p_value': p_value_ks
                    }

    return results


def plot_error_distributions(ticker_results, ticker):
    """Plot error distributions for different feature combinations."""
    error_distributions = {}

    for variation, data in ticker_results.items():
        if 'metrics' in data and 'test_predictions' in data['metrics'] and 'test_actual' in data['metrics']:
            errors = calculate_errors(
                data['metrics']['test_predictions'], data['metrics']['test_actual'])
            error_distributions[variation] = errors

    if not error_distributions:
        print(f"No valid error distributions found for {ticker}")
        return

    plt.figure(figsize=(12, 6))
    for variation, errors in error_distributions.items():
        sns.kdeplot(errors, label=variation)

    plt.title(f'Error Distributions for {ticker}')
    plt.xlabel('Absolute Prediction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(
        f'results/statistical_analysis/{ticker}_error_distributions.png')
    plt.close()


def save_results_to_csv(all_results, output_dir):
    """Save statistical test results to CSV files."""
    # Prepare data for CSV
    kw_data = []
    ks_data = []
    error_data = []

    for ticker, results in all_results.items():
        # Kruskal-Wallis test results
        for group_name, group_results in results['group_tests'].items():
            kw_data.append({
                'Ticker': ticker,
                'Group': group_name,
                'Variations': ', '.join(group_results['variations']),
                'H_Statistic': group_results['kruskal_wallis']['h_statistic'],
                'P_Value': group_results['kruskal_wallis']['p_value']
            })

        # KS test results
        for group_name, group_results in results['group_tests'].items():
            for comparison, test_results in group_results['ks_tests'].items():
                ks_data.append({
                    'Ticker': ticker,
                    'Group': group_name,
                    'Comparison': comparison,
                    'KS_Statistic': test_results['ks_statistic'],
                    'P_Value': test_results['p_value']
                })

        # Error statistics
        for variation, mean in results['error_means'].items():
            error_data.append({
                'Ticker': ticker,
                'Variation': variation,
                'Mean_Error': mean,
                'Std_Error': results['error_stds'][variation]
            })

    # Convert to DataFrames and save
    pd.DataFrame(kw_data).to_csv(
        output_dir / 'kruskal_wallis_results.csv', index=False)
    pd.DataFrame(ks_data).to_csv(
        output_dir / 'ks_test_results.csv', index=False)
    pd.DataFrame(error_data).to_csv(
        output_dir / 'error_statistics.csv', index=False)


def main():
    # Create output directory
    output_dir = Path('results/statistical_analysis')
    output_dir.mkdir(exist_ok=True)

    # List of tickers to analyze
    tickers = ['AMD', 'INTC', 'JPM', 'BAC', 'JNJ', 'PFE']

    all_results = {}

    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")

        # Load results
        ticker_results = load_lstm_results(ticker)
        if ticker_results is None:
            continue

        # Perform statistical tests
        test_results = perform_statistical_tests(ticker_results)
        if test_results is None:
            continue

        all_results[ticker] = test_results

        # Plot error distributions
        plot_error_distributions(ticker_results, ticker)

        # Print results
        print(f"\nResults for {ticker}:")

        print("\nGroup Tests:")
        for group_name, group_results in test_results['group_tests'].items():
            print(f"\n{group_name}:")
            print("Variations:", group_results['variations'])
            print("Kruskal-Wallis Test:")
            print(
                f"H-statistic: {group_results['kruskal_wallis']['h_statistic']:.4f}")
            print(f"p-value: {group_results['kruskal_wallis']['p_value']:.4f}")

            print("\nKS Test Results:")
            for comparison, results in group_results['ks_tests'].items():
                print(f"{comparison}:")
                print(f"  KS-statistic: {results['ks_statistic']:.4f}")
                print(f"  p-value: {results['p_value']:.4f}")

        print("\nError Statistics:")
        for variation, mean in test_results['error_means'].items():
            std = test_results['error_stds'][variation]
            print(f"{variation}: Mean = {mean:.4f}, Std = {std:.4f}")

    # Save all results
    with open(output_dir / 'statistical_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    # Save results to CSV files
    save_results_to_csv(all_results, output_dir)


if __name__ == "__main__":
    main()
