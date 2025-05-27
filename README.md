# Analysis of Information Flow Impact on Stock Prices

This project analyzes how news sentiment affects stock prices across three major sectors: Technology (AMD, Intel), Banking (JPM, BAC), and Healthcare (PFE, JNJ). The analysis combines sentiment analysis of news headlines and summaries with stock price movements to understand information flow impacts.

## Methodology

- **Sentiment Analysis**: Using VADER sentiment analyzer to score news headlines and summaries
- **Statistical Measures**: MSE, RMSE, MAE, and Kolmogorov-Smirnov tests
- **Correlation Analysis**: Pearson correlation between sentiment and stock returns
- **Clickbait Detection**: Analysis of discrepancies between headline and summary sentiments

## Results by Sector

### 1. Technology Sector (AMD vs Intel)

#### News Coverage
- AMD: 1,080 articles (1.42 articles/day)
- Intel: 1,182 articles (1.42 articles/day)
- Highest news coverage among all sectors, indicating high media attention

#### Sentiment Impact
- Intel shows significant correlation between summary sentiment and returns (p-value: 0.0166)
- AMD shows weaker correlation (-0.0009 for headlines, 0.0118 for summaries)
- Both companies show high sentiment volatility (RMSE: AMD 0.191, Intel 0.192)

#### CPU Competition Analysis
- Specific focus on CPU-related news and product launches
- Timeline analysis of major events:
  - Ryzen 7000 Launch Period (2022-09 to 2022-12)
  - X3D Launch Period (2023-02 to 2023-05)
- Relative stock performance comparison available in `results/competition/`

### 2. Banking Sector (JPM vs BAC)

#### News Coverage
- JPM: 360 articles (1.20 articles/day)
- BAC: 430 articles (1.23 articles/day)
- More balanced and moderate news coverage compared to tech sector

#### Sentiment Impact
- JPM shows interesting contrast between headline (0.0255) and summary (-0.0359) sentiment correlation
- BAC displays minimal correlation (0.0139 headlines, -0.0003 summaries)
- Lower sentiment volatility compared to tech sector (RMSE: JPM 0.135, BAC 0.134)

#### Key Findings
- Banking sector shows more stable sentiment patterns
- Lower clickbait tendency (smaller headline-summary discrepancies)
- More consistent sentiment across headlines and summaries

### 3. Healthcare Sector (PFE vs JNJ)

#### News Coverage
- PFE: 447 articles (1.15 articles/day)
- JNJ: 321 articles (1.14 articles/day)
- Lowest news coverage among sectors

#### Sentiment Impact
- PFE shows stronger headline sentiment correlation (0.0328) than JNJ (0.0034)
- JNJ displays more balanced sentiment impact between headlines and summaries
- Lowest sentiment volatility among sectors (RMSE: JNJ 0.128, PFE 0.147)

#### Key Findings
- Most stable sentiment patterns among all sectors
- Lowest clickbait scores
- Most consistent correlation between headlines and summaries

## Statistical Significance

### Kolmogorov-Smirnov Test Results
All companies show significant differences between sentiment and returns distributions (p-values < 0.05):
- Tech Sector: Strongest deviation (AMD: 0.4076, Intel: 0.3628)
- Banking Sector: Moderate deviation (JPM: 0.4614, BAC: 0.4446)
- Healthcare Sector: Highest deviation (PFE: 0.4745, JNJ: 0.4496)

### Mean Absolute Error (MAE)
Sector comparison of headline sentiment accuracy:
1. Healthcare: 0.042-0.056
2. Banking: 0.047-0.052
3. Technology: 0.099-0.101

## Key Insights

1. **Sector-Specific Patterns**
   - Technology: Highest news volume, most volatile sentiment
   - Banking: Moderate coverage, stable sentiment
   - Healthcare: Lowest coverage, most consistent sentiment

2. **Clickbait Analysis**
   - Tech sector shows highest headline-summary discrepancies
   - Healthcare shows most consistent headline-summary alignment
   - Banking sector shows moderate discrepancies

3. **Information Flow Impact**
   - Immediate impact strongest in tech sector
   - Most stable impact in healthcare sector
   - Banking sector shows moderate, consistent impact

## Detailed Results

Detailed analysis results are available in the following directories:
- `results/sentiment/`: Individual company sentiment analysis
- `results/competition/`: AMD vs Intel specific analysis
- `results/plots/`: Visualization of sentiment trends and correlations

## Conclusions

1. **Sector Characteristics**
   - Technology sector shows highest sensitivity to news sentiment
   - Healthcare sector shows most stable relationship with news sentiment
   - Banking sector shows moderate sensitivity with consistent patterns

2. **Information Processing**
   - Different sectors show distinct patterns in how information is processed
   - Tech sector shows fastest reaction to news
   - Healthcare shows most measured response
   - Banking shows consistent moderate response

3. **Investment Implications**
   - Consider sector-specific news impact patterns in trading strategies
   - Account for higher volatility in tech sector sentiment
   - Consider longer-term sentiment trends in healthcare sector
   - Focus on summary sentiment over headlines in banking sector

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

## Data Collection
- Headlines and first paragraphs from Seeking Alpha
- Historical stock prices
- Time period: 2019-2024

## Models
- LSTM for main prediction
- KNN as baseline
- Sentiment analysis for news content 