# News Sentiment Impact on Stock Prices Analysis

## Author
Kuzakhmetov Ilmir Rinatovich

## Project Overview
This project analyzes the relationship between news sentiment and stock price movements for major companies. It employs various machine learning models and statistical analysis techniques to understand how news sentiment affects stock returns and to predict potential price movements based on news content.

## Key Features
- Sentiment analysis of news headlines and summaries using VADER sentiment analyzer
- Multiple machine learning approaches:
  - K-Nearest Neighbors (KNN) model
  - Long Short-Term Memory (LSTM) neural network
- Comparative analysis across different sectors
- Competitor analysis within the same industry
- Clickbait detection and filtering
- Comprehensive statistical analysis including correlation, MSE, RMSE, and MAE metrics

## Analyzed Companies
The project analyzes the following companies:
- AMD (Advanced Micro Devices)
- BAC (Bank of America)
- INTC (Intel Corporation)
- JNJ (Johnson & Johnson)
- JPM (JPMorgan Chase)
- PFE (Pfizer)

## Project Structure
```
.
├── data/                  # Data directory containing stock and news data
├── results/              # Analysis results and visualizations
│   ├── vader_analysis/   # VADER sentiment analysis results
│   ├── knn_analysis/     # KNN model results
│   └── lstm_analysis/    # LSTM model results
├── src/                  # Source code
│   ├── analysis/         # Analysis modules
│   ├── features/         # Feature engineering
│   ├── models/          # ML model implementations
│   └── main.py          # Main execution script
├── requirements.txt      # Project dependencies
└── statistical_analysis.py # Statistical analysis scripts
```

## Dependencies
The project requires Python 3.x and the following main packages:
- pandas >= 2.1.0
- numpy >= 1.26.0
- matplotlib >= 3.8.0
- seaborn >= 0.13.0
- tensorflow >= 2.15.0
- scikit-learn >= 1.3.0
- nltk >= 3.8.1
- scipy >= 1.11.0
- plotly >= 5.18.0
- python-dotenv >= 1.0.0
- requests >= 2.31.0
- beautifulsoup4 >= 4.12.2

## Installation
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure your data files are placed in the `data/` directory:
   - `stock_data.csv`: Historical stock price data
   - `{SYMBOL}_news.csv`: News data for each company (e.g., `AMD_news.csv`)

2. Run the main analysis:
   ```bash
   python src/main.py
   ```

The script will:
- Load and preprocess the data
- Perform sentiment analysis on news headlines and summaries
- Run KNN and LSTM models with various feature combinations
- Generate visualizations and statistical analysis
- Save results in the `results/` directory

## Analysis Components
1. **Sentiment Analysis**
   - VADER sentiment analysis on news headlines and summaries
   - Daily sentiment aggregation
   - Correlation analysis with stock returns

2. **Machine Learning Models**
   - KNN model variations:
     - Price-only predictions
     - Price + headline sentiment
     - Price + summary sentiment
     - Combined features
     - Clickbait-filtered variations
   - LSTM model variations with similar feature combinations

3. **Statistical Analysis**
   - Correlation analysis
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - Kolmogorov-Smirnov tests

## Results
Results are organized in the `results/` directory:
- `vader_analysis/`: Sentiment analysis results and visualizations
- `knn_analysis/`: KNN model predictions and performance metrics
- `lstm_analysis/`: LSTM model predictions and performance metrics

Each analysis type includes:
- Performance metrics
- Visualization plots
- Statistical test results
- Model predictions
