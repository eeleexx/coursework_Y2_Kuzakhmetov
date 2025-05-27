# Analysis of Information Flow Impact on Stock Prices

This project analyzes how news sentiment affects stock prices across three major sectors: Technology (AMD, Intel), Banking (JPM, BAC), and Healthcare (PFE, JNJ). The analysis combines sentiment analysis, machine learning, and deep learning approaches to understand and predict stock movements based on news and technical indicators.

## Analysis Methods

The project employs three complementary analysis approaches:

### 1. VADER Sentiment Analysis Results
*Baseline sentiment analysis to understand news impact*

#### Overall Performance
- Successfully analyzed over 3,800 news articles across 6 stocks
- Captured both headline and summary sentiments
- Identified sector-specific sentiment patterns

#### Sector-Specific Findings

##### Technology Sector (AMD, Intel)
- Highest news coverage: ~1.42 articles/day
- Intel: Significant correlation with returns (p-value: 0.0166)
- AMD: Weaker correlation (-0.0009 headlines, 0.0118 summaries)
- High sentiment volatility (RMSE: AMD 0.191, Intel 0.192)

##### Banking Sector (JPM, BAC)
- Moderate news coverage: ~1.21 articles/day
- JPM: Contrasting headline (0.0255) vs summary (-0.0359) correlation
- BAC: Minimal correlation (0.0139 headlines, -0.0003 summaries)
- Lower sentiment volatility (RMSE: JPM 0.135, BAC 0.134)

##### Healthcare Sector (PFE, JNJ)
- Lowest news coverage: ~1.14 articles/day
- Most stable sentiment patterns
- Lowest clickbait scores
- PFE: Stronger headline correlation (0.0328)
- JNJ: More balanced sentiment impact
- Lowest volatility (RMSE: JNJ 0.128, PFE 0.147)

#### Statistical Significance
- All stocks show significant KS test results (p < 0.05)
- Sector-specific MAE patterns:
  - Healthcare: 0.042-0.056
  - Banking: 0.047-0.052
  - Technology: 0.099-0.101

### 2. KNN Analysis Results
*Pattern-matching approach for movement prediction*

#### Overall Performance
- Training accuracy: 68-70% across all stocks
- Test accuracy: 48-54% indicating overfitting
- Best performers: JPM (53.1%) and PFE (54.3%)

#### Stock-Specific Performance

##### Technology Sector
- AMD:
  - Training: 69.3% accuracy, 69.5% F1
  - Test: 50.6% accuracy, 54.4% F1
  - Key features: price/MA ratios

- INTC:
  - Training: 70.0% accuracy, 70.9% F1
  - Test: 49.1% accuracy, 47.4% F1
  - Negative importance for volatility

##### Banking Sector
- JPM:
  - Training: 70.3% accuracy, 72.0% F1
  - Test: 53.1% accuracy, 56.0% F1
  - Most balanced performance

- BAC:
  - Training: 69.1% accuracy, 69.8% F1
  - Test: 49.7% accuracy, 51.5% F1
  - News volume important

##### Healthcare Sector
- PFE:
  - Training: 69.2% accuracy, 66.3% F1
  - Test: 54.3% accuracy, 51.8% F1
  - Returns most important

- JNJ:
  - Training: 68.3% accuracy, 69.8% F1
  - Test: 48.1% accuracy, 46.0% F1
  - Negative sentiment importance

#### Feature Importance Findings
- Technical indicators generally more important than sentiment
- News volume shows moderate importance
- Stock-specific feature importance patterns
- Price momentum and MA ratios consistently important

#### Model Characteristics
- Consistent ~20% performance drop from training to test
- Better at capturing technical patterns than sentiment impact
- Shows sector-specific prediction patterns

### 3. LSTM Analysis Results
*Deep learning approach for sequence prediction*

#### Model Architecture
- Sequence-based prediction
- Combined technical and sentiment features
- Multi-step prediction capability

#### Current Status
- Model implementation in progress
- Initial testing phase
- Preliminary results pending

## Current Insights (VADER + KNN)

1. **Sector-Specific Patterns**
   - Technology: Highest volatility, strongest news impact
   - Banking: Most balanced performance
   - Healthcare: Most stable patterns

2. **Prediction Challenges**
   - Short-term prediction remains difficult
   - Technical indicators more reliable than sentiment
   - Sector-specific approaches needed

3. **Information Processing**
   - Different sectors show distinct patterns
   - Tech sector most sensitive to news
   - Healthcare shows most stable response
   - Banking shows consistent moderate impact

## Project Structure

### Data Collection
- Headlines and first paragraphs from Seeking Alpha
- Historical stock prices
- Time period: 2019-2024

### Analysis Pipeline
1. VADER Sentiment Analysis
   - News sentiment scoring
   - Statistical analysis
   - Sector comparisons

2. KNN Analysis
   - Feature engineering
   - Pattern matching
   - Performance evaluation

3. LSTM Analysis (In Progress)
   - Sequence modeling
   - Combined feature analysis
   - Multi-step prediction

## Setup
1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Results Directory Structure
- `results/vader_analysis/`: Sentiment analysis results
- `results/knn_analysis/`: KNN model results and visualizations
- `results/lstm_analysis/`: LSTM analysis (upcoming) 