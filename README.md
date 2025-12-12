# Provincial Economic ML Analysis - Nepal
## Machine Learning-Based Classification and Clustering of Provincial Economies

**Author:** Smaran Luitel
**Course:** STW7072CEM - Machine Learning
**Institution:** Softwarica College / Coventry University
**Date:** December 2025

---

## 📋 Project Overview

This project applies machine learning techniques to analyze Nepal's 7 provincial economies using actual National Accounts data. The analysis employs both unsupervised (clustering) and supervised (classification) learning to identify economic patterns and predict development tiers.

### Key Results:
- ✅ **Clustering**: k=2 optimal clusters (Silhouette=0.502)
- ✅ **Classification**: 71.4% accuracy (Random Forest), 57.1% (Gradient Boosting)
- ✅ **Method Agreement**: Perfect consensus (ARI=1.000) between K-means and Hierarchical clustering
- ✅ **Key Finding**: Bagmati province economically dominant, forms separate cluster

---

## 🎯 Project Objectives

1. Develop ML framework to identify economic clusters among Nepal's 7 provinces
2. Build predictive models to classify provinces into development tiers
3. Identify key economic indicators driving provincial differences
4. Provide data-driven insights for federal economic policy

---

## 📊 Dataset

**Source:** National Accounts of Nepal - Provincial GDP Dataset

**Specifications:**
- **Provinces (7)**: Koshi, Madhesh, Bagmati, Gandaki, Lumbini, Karnali, Sudurpashchim
- **Economic Sectors (17)**: A-Q (NSIC classification)
- **Time Period**: 7 years (2075/76 to 2081/82 Nepali calendar)
- **Data Points**: ~3,000 observations
- **File**: `data/National-Accounts.xlsx`

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- 8GB RAM (recommended)
- Windows/Mac/Linux

### Step 1: Clone or Download Project
```bash
https://github.com/smrnltl/ML-assignment
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- pandas (≥1.5.0)
- numpy (≥1.24.0)
- scikit-learn (≥1.2.0)
- xgboost (≥1.7.0)
- matplotlib (≥3.7.0)
- seaborn (≥0.12.0)
- scipy (≥1.10.0)
- openpyxl (≥3.1.0)
- jupyter (≥1.0.0)

### Step 3: Verify Installation
```bash
python -c "import pandas, numpy, sklearn, xgboost, matplotlib; print('All packages installed!')"
```

---

## 🚀 Usage

### Option 1: Jupyter Notebook 

```bash
jupyter notebook provincial_ml_pipeline.ipynb
```

**Then:**
1. Click "Kernel" → "Restart & Run All"
2. Wait 2-3 minutes for complete execution
3. Review inline visualizations and results

### Option 2: Python Script (For Testing)

```bash
python provincial_ml_pipeline.py
```

**Output:** All results saved to `outputs/` directory

---

## 📁 Project Structure

```
project/
├── provincial_ml_pipeline.ipynb        # Main Jupyter Notebook
├── provincial_ml_pipeline.py           # Complete Python implementation
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
│
├── data/
│   └── National-Accounts.xlsx         # Source data
│
└── outputs/                           # Generated results
    ├── optimal_clusters_analysis.png          # Clustering optimization
    ├── hierarchical_dendrogram.png            # Province relationships
    ├── clustering_agreement.png               # Method comparison
    ├── pca_clusters_comparison.png            # PCA visualization
    ├── tsne_visualization.png                 # t-SNE visualization
    ├── comprehensive_dashboard.png            # 9-panel summary
    ├── provincial_analysis_results.csv        # Complete results table
    └── analysis_summary.json                  # Key metrics
```

---

## 🔬 Methodology

### 1. Data Extraction
- Loads Provincial GDP sheet from National Accounts
- Extracts 17 economic sectors for 7 provinces
- Averages 7 years of data for stability

### 2. Feature Engineering
**Created 50 features from 17 original sectors:**
- **Diversity Indices**: HHI, Shannon Entropy, Gini Coefficient
- **Structural Indicators**: Primary/Secondary/Tertiary shares
- **Modernization Index**: (Secondary + Tertiary) / Total GDP
- **Statistical Measures**: Variance, Skewness, Kurtosis
- **Relative Performance**: Province-to-national ratios

### 3. Clustering Analysis (Unsupervised Learning)
**2 Core Algorithms Applied:**
- **K-means**: Centroid-based partitioning
- **Hierarchical**: Agglomerative Ward linkage

**Additional algorithms (GMM, DBSCAN) were implemented for validation but not used in core analysis.**

**Validation Metrics:**
- Silhouette Score (higher = better separation)
- Davies-Bouldin Score (lower = better compactness)
- Calinski-Harabasz Score (higher = better definition)
- Adjusted Rand Index (inter-method agreement)

### 4. Classification Analysis (Supervised Learning)
**2 Core Classifiers Trained:**
- **Random Forest**: Ensemble of 100 decision trees (71.4% accuracy)
- **Gradient Boosting**: Sequential error correction (57.1% accuracy)

**XGBoost was also implemented but overfitted on n=7 (14.3% accuracy), so it's not part of the core analysis.**

**Validation Strategy:**
- Leave-One-Out Cross-Validation (LOO-CV)
- Appropriate for small sample size (n=7)
- Feature selection: 50 → 10 most important features
- Shallow trees (max_depth=3) to prevent overfitting

### 5. Visualization
- PCA: Linear dimensionality reduction (50D → 2D)
- t-SNE: Non-linear dimensionality reduction
- Dendrograms: Hierarchical relationships
- Comprehensive dashboard: 9-panel summary

---

## 📈 Key Results

### Clustering Results

| Metric | Value |
|--------|-------|
| Optimal Clusters (k) | 2 |
| Silhouette Score | 0.502 |
| Davies-Bouldin Score | 0.278 |
| Method Agreement (ARI) | 1.000 |

**Cluster Composition:**
- **Cluster 0**: Bagmati (alone)
  - GDP: 1,550,187 (3.4x larger than average)
  - Modernization Index: 0.772 (77.2% non-primary sectors)

- **Cluster 1**: Koshi, Madhesh, Gandaki, Lumbini, Karnali, Sudurpashchim
  - Average GDP: 449,964
  - Modernization Index: 0.411

### Classification Results

| Classifier | LOO-CV Accuracy |
|------------|----------------|
| **Random Forest** | **71.4%** |
| Gradient Boosting | 57.1% |

**Development Tier Assignment:**
- **Advanced (2)**: Koshi, Bagmati
- **Emerging (3)**: Madhesh, Gandaki, Lumbini
- **Lagging (2)**: Karnali, Sudurpashchim

**Top Predictive Features:**
1. Sector L (Real Estate) - Relative performance
2. Sector N (Administrative Services) - Relative performance
3. Sector J (Information/Communication)

---

## 🎓 Academic Contributions

### Methodological
- Framework for ML application in small federal systems (n=7)
- Demonstrates Leave-One-Out CV for small sample classification
- Multi-algorithm validation strategy with consensus checking

### Empirical
- First ML-based classification of Nepal's provincial economies
- Quantitative validation of Bagmati's economic dominance
- Identification of service sectors as key development predictors

### Policy-Relevant
- Objective provincial categorization for targeted interventions
- Evidence-based framework for federal transfer mechanisms
- Reproducible methodology for tracking provincial evolution

---

## 💡 Policy Implications

1. **Two-Tier Development Strategy**
   - Bagmati requires different development approach
   - Other 6 provinces can share similar programs

2. **Infrastructure Priority**
   - Real estate and administrative services highly predictive
   - Service sector modernization critical for advancement

3. **Targeted Support**
   - Karnali and Sudurpashchim need immediate intervention
   - Bagmati can serve as economic growth driver for region

---

## ⚠️ Limitations

1. **Small Sample Size**: n=7 limits statistical power
2. **Temporal Averaging**: 7-year average may mask trends
3. **Cross-Sectional**: No time-series dynamics captured
4. **Data Limitations**: Relies solely on National Accounts data

---

## 🔮 Future Work

1. **Temporal Analysis**: Time-series clustering of provincial evolution
2. **Network Analysis**: Model inter-provincial economic dependencies
3. **External Features**: Include demographic, geographic data
4. **Predictive Modeling**: Forecast development trajectories
5. **Causality Analysis**: Identify drivers of economic transitions

---

### To Run Before Submission:
1. Restart kernel: `Kernel → Restart & Clear Output`
2. Run all cells: `Kernel → Restart & Run All`
3. Verify all visualizations display
4. Check outputs/ directory has 8 files
5. Save notebook with outputs: `File → Save`

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'xgboost'"
**Solution:**
```bash
pip install xgboost
```

### Issue: "FileNotFoundError: data/National-Accounts.xlsx"
**Solution:** Ensure data file is in `data/` subdirectory

### Issue: Visualizations not displaying in notebook
**Solution:**
```python
# Add to cell:
%matplotlib inline
from IPython.display import Image
```

### Issue: "MemoryError"
**Solution:** Close other applications, ensure 8GB+ RAM available

---

## 📚 References

1. Nepal Rastra Bank (2023). *National Accounts Statistics of Nepal*. Kathmandu: NRB.
2. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD'16*.
4. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation. *Journal of Computational and Applied Mathematics*, 20, 53-65.

---

## 📧 Contact

**Student:** Smaran Luitel
**Student ID:** 250087
**Email:** 250087@softwarica.edu.np
**Module:** STW7072CEM - Machine Learning
**Submission Date:** December 10, 2025

---

## 📄 License

This project is submitted as coursework for STW7072CEM. All code and analysis are original work unless cited.

---

## 🙏 Acknowledgments

- Nepal Rastra Bank for providing National Accounts data
- Softwarica College / Coventry University for academic support
- Module Leader: Shrawan Thakur

---

**Last Updated:** December 10, 2024
**Version:** 1.0 (Final Submission)
