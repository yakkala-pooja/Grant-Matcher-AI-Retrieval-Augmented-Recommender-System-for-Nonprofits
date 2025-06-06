# GrantMatch AI

An intelligent grant matching and nonprofit risk assessment system that uses advanced NLP and anomaly detection to connect nonprofits with relevant grants while identifying potential risks and anomalies.

## 🎯 Overview

GrantMatch AI is a comprehensive platform that combines grant recommendation with fraud detection and data quality assessment. The system processes nonprofit data to:
1. Match organizations with relevant grants using semantic similarity
2. Detect anomalous patterns in financial and impact data
3. Assess data quality and completeness
4. Provide a user-friendly dashboard for exploring recommendations and alerts

## 📊 Project Statistics
- **Total Lines of Code**: 1,232
- **Python Files**: 1,232 lines
  - data_loader.py: 183 lines
  - grant_recommender.py: 235 lines
  - app.py: 316 lines
  - main.py: 171 lines
  - generate_final_datasets.py: 326 lines
- **Core Components**: 5 Python modules
- **Data Processing**: 240,585 nonprofit records analyzed

## 🚀 Features

### Grant Matching
- Semantic similarity-based grant recommendations
- Customizable matching weights for different factors
- Mission statement analysis
- Financial compatibility assessment
- Impact score consideration

### Risk Assessment
- Multi-criteria anomaly detection
  - High impact efficiency outliers
  - Extreme revenue patterns
  - Impact score mismatches
  - Suspicious financial patterns
- Data quality scoring
- Comprehensive risk levels (Critical, High, Medium, Low)

### Dashboard
- Interactive Streamlit interface
- Real-time grant matching
- Fraud alert visualization
- Configurable model settings
- Detailed nonprofit profiles

## 🛠 Technical Architecture

### Core Components
1. `app.py` (316 lines)
   - Streamlit dashboard implementation
   - User interface and visualization
   - Real-time data filtering and display

2. `generate_final_datasets.py` (326 lines)
   - Data processing pipeline
   - Anomaly detection algorithms
   - Synthetic data generation
   - Quality assessment

3. `grant_recommender.py` (235 lines)
   - FAISS-based recommendation engine
   - Semantic similarity computation
   - Model persistence management

4. `data_loader.py` (183 lines)
   - Data ingestion and preprocessing
   - File format handling
   - Data validation

5. `main.py` (171 lines)
   - Application entry point
   - System coordination
   - Error handling

### Data Files
- `non-profits_final.csv`: Processed nonprofit data
- `grants_final.csv`: Grant opportunity database
- `funder_history.json`: Historical funding records
- `nonprofit_anomalies.csv`: Detected anomalies
- `nonprofit_quality.csv`: Data quality metrics

## 🔄 Pipeline Summary

The system's data processing and recommendation pipeline flows through multiple Python files:

1. **Initial Data Processing** (`generate_final_datasets.py`):
   - Loads raw nonprofit and grant data
   - Processes and cleans the data
   - Generates synthetic data where needed
   - Creates:
     - `non-profits_final.csv`
     - `grants_final.csv`
     - `nonprofit_quality.csv`
     - `nonprofit_anomalies.csv`
     - `funder_history.json`

2. **Data Loading Layer** (`data_loader.py`):
   - Handles all data ingestion operations
   - Provides data cleaning and preprocessing functions
   - Manages funder history data
   - Creates nonprofit-funder mappings
   - Converts data formats (dates, currency, etc.)

3. **Recommendation Engine** (`grant_recommender.py`):
   - Implements FAISS-based semantic search
   - Manages model weights and embeddings
   - Processes mission statements
   - Generates match justifications
   - Incorporates funder history for ranking
   - Handles model persistence

4. **Application Core** (`main.py`):
   - Coordinates the overall system
   - Manages recommendation workflow
   - Handles data formatting and display
   - Processes nonprofit information
   - Formats recommendations output

5. **User Interface** (`app.py`):
   - Implements Streamlit dashboard
   - Manages user interactions
   - Displays grant matches
   - Shows fraud alerts
   - Handles model settings
   - Visualizes nonprofit data

### Data Flow:
```
[Raw Data] → generate_final_datasets.py
    ↓
[Processed Data] → data_loader.py
    ↓
[Clean, Structured Data] → grant_recommender.py
    ↓
[Recommendations] → main.py
    ↓
[User Interface] → app.py
```

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GrantMatch_AI.git
cd GrantMatch_AI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚦 Getting Started

1. Generate the datasets:
```bash
python generate_final_datasets.py
```

2. Launch the dashboard:
```bash
streamlit run app.py
```

## 💡 Usage

### Grant Matching
1. Navigate to the "Grant Matching" tab
2. Select a nonprofit organization
3. View organization details and impact metrics
4. Click "Find Matching Grants" to get recommendations

### Risk Assessment
1. Go to the "Fraud Alerts" tab
2. Use filters to focus on specific risk levels
3. Examine the anomaly visualizations
4. Review detailed nonprofit profiles

### Model Configuration
1. Access the "Model Settings" tab
2. Adjust matching weights
3. Save and visualize new configurations

## 🔍 Anomaly Detection System

The system uses multiple criteria to identify potential risks:

1. Impact Efficiency Analysis
   - Uses IQR-based outlier detection
   - Considers revenue-to-impact ratios
   - Identifies suspicious patterns

2. Revenue Pattern Analysis
   - Detects extreme outliers (±3 standard deviations)
   - Identifies suspicious zero-revenue cases
   - Analyzes asset-to-revenue ratios

3. Impact Score Validation
   - Checks for mismatches with financial data
   - Validates against expected percentiles
   - Identifies inconsistent claims

4. Risk Scoring
   - Weighted combination of multiple factors
   - Comprehensive risk level assignment
   - Detailed anomaly categorization

## 📈 Development History

### Initial Development
- Processed 240,585 nonprofit records
- Implemented basic impact score assignment
- Created synthetic test cases

### Enhancements
- Added multi-criteria anomaly detection
- Improved data quality assessment
- Implemented confidence scoring
- Enhanced visualization capabilities

### Optimizations
- Modified impact efficiency calculations
- Added logarithmic scaling
- Improved anomaly type classification
- Enhanced risk level granularity

## 🤖 Agent Mode Usage

The project was developed with the assistance of an AI agent. Here are the key interactions:

1. Initial Data Processing Query:
   - Processed nonprofit data (240,585 records)
   - Assigned impact scores
   - Created synthetic test cases

2. Anomaly Detection Development:
   - Debugged NaN values in IQR calculations
   - Modified confidence thresholds
   - Implemented multiple detection criteria

3. Test Case Creation:
   - Created various synthetic anomaly cases
   - Tested edge cases and extreme scenarios
   - Validated detection accuracy

4. Dashboard Implementation:
   - Developed interactive visualizations
   - Implemented filtering systems
   - Added detailed nonprofit profiles

5. System Optimization:
   - Enhanced risk level display
   - Improved data loading efficiency
   - Optimized anomaly detection algorithms

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.