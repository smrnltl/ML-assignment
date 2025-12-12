#!/usr/bin/env python3
"""
Provincial Economic ML Analysis Pipeline for Nepal
==================================================
Author: Smaran Luitel
Date: Dec 2025
"""

import pandas as pd
import numpy as np
import warnings
import json
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Create output directory for results
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Core ML Libraries
from sklearn.cluster import KMeans, AgglomerativeClustering  # Removed DBSCAN - not used in core analysis
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    LeaveOneOut, GridSearchCV, cross_validate
)
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, make_scorer
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline

# Advanced ML
from sklearn.manifold import TSNE
# Removed: from sklearn.mixture import GaussianMixture - not used in core analysis
# Removed: import xgboost as xgb - overfits on n=7

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist
from scipy import stats

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# ==========================================
# PART 1: DATA EXTRACTION AND PREPROCESSING
# ==========================================

class ProvincialDataExtractor:
    """Extract and structure provincial GDP data from National Accounts"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        # Province names in the ORDER they appear in the Excel file
        self.province_names = [
            'Koshi', 'Madhesh', 'Bagmati', 'Gandaki',
            'Lumbini', 'Karnali', 'Sudurpashchim'
        ]
        self.sectors = {}
        self.raw_data = None
        self.structured_data = None
        
    def load_data(self):
        """Load and parse the Provincial GDP sheet"""
        print("Loading Provincial GDP data from National Accounts...")
        
        # Read the raw data
        df = pd.read_excel(self.file_path, sheet_name='Provincial GDP', header=None)
        self.raw_data = df
        
        print(f"Raw data shape: {df.shape}")
        
        # Find province columns and years
        self._identify_structure()
        
        # Extract sector codes and descriptions
        self._extract_sectors()
        
        # Create structured dataset
        self._create_structured_data()
        
        return self.structured_data
    
    def _identify_structure(self):
        """Identify the structure of the data sheet"""
        df = self.raw_data
        
        # Find row with years (contains 2075/76, 2076/77, etc.)
        self.year_row = None
        for i in range(10):
            row = df.iloc[i]
            if any('2075/76' in str(v) or '2076/77' in str(v) for v in row.values if pd.notna(v)):
                self.year_row = i
                break
        
        # Find row with province names
        self.province_row = None
        for i in range(10):
            row = df.iloc[i]
            if any('Province' in str(v) or 'Madhes' in str(v) or 'Bagmati' in str(v) 
                   for v in row.values if pd.notna(v)):
                self.province_row = i
                break
        
        # Data typically starts after the header rows
        self.data_start_row = max([r for r in [self.year_row, self.province_row] if r is not None]) + 2
        
        print(f"Year row: {self.year_row}")
        print(f"Province row: {self.province_row}")
        print(f"Data starts at row: {self.data_start_row}")
    
    def _extract_sectors(self):
        """Extract sector codes and descriptions"""
        df = self.raw_data
        
        # Sectors are typically in first two columns
        for i in range(self.data_start_row, min(self.data_start_row + 30, len(df))):
            code = df.iloc[i, 0]
            description = df.iloc[i, 1]
            
            if pd.notna(code) and isinstance(code, str) and len(code) <= 2:
                if pd.notna(description):
                    self.sectors[code] = str(description)
        
        print(f"Found {len(self.sectors)} economic sectors")
        print(f"Sectors: {list(self.sectors.keys())}")
    
    def _create_structured_data(self):
        """Create structured dataset from raw data"""
        df = self.raw_data
        
        # Find numeric columns (GDP values)
        numeric_cols = []
        for col in range(2, df.shape[1]):  # Start from column 2
            # Check if column has numeric data
            sample = pd.to_numeric(df.iloc[self.data_start_row:self.data_start_row+10, col], 
                                  errors='coerce')
            if sample.notna().sum() > 5:
                numeric_cols.append(col)
        
        print(f"Found {len(numeric_cols)} numeric columns")
        
        # Determine years from row headers
        years = []
        if self.year_row is not None:
            for col in numeric_cols:
                year_val = df.iloc[self.year_row, col]
                if pd.notna(year_val):
                    years.append(str(year_val).strip())
        
        # Create structured data
        structured_data = []
        
        # Estimate provinces per year (7 provinces)
        n_provinces = 7
        cols_per_year = len(numeric_cols) // 8  # 8 years of data expected
        
        # Extract data for each province
        for prov_idx, province in enumerate(self.province_names):
            province_data = {'Province': province}

            # Calculate column range for this province
            # Each province has 7 years of data (2075/76 to 2081/82)
            start_col = 2 + (prov_idx * 7)  # Each province has 7 years
            end_col = start_col + 7
            
            # Get sector values (average across years for stability)
            for sector_code in self.sectors.keys():
                # Find row for this sector
                sector_row = None
                for i in range(self.data_start_row, min(self.data_start_row + 30, len(df))):
                    if df.iloc[i, 0] == sector_code:
                        sector_row = i
                        break
                
                if sector_row is not None:
                    # Get values for this province across years
                    values = []
                    for col in range(start_col, min(end_col, df.shape[1])):
                        val = pd.to_numeric(df.iloc[sector_row, col], errors='coerce')
                        if pd.notna(val):
                            values.append(val)
                    
                    # Store average value
                    if values:
                        province_data[f'Sector_{sector_code}'] = np.mean(values)
                    else:
                        province_data[f'Sector_{sector_code}'] = 0
            
            structured_data.append(province_data)
        
        # Convert to DataFrame
        self.structured_data = pd.DataFrame(structured_data)
        self.structured_data.set_index('Province', inplace=True)
        
        print(f"Structured data shape: {self.structured_data.shape}")
        print(f"Provinces: {list(self.structured_data.index)}")
        print(f"Features: {list(self.structured_data.columns[:5])}...")
        
        return self.structured_data

# ==========================================
# PART 2: FEATURE ENGINEERING
# ==========================================

class FeatureEngineer:
    """Advanced feature engineering for provincial economic data"""
    
    def __init__(self, data):
        self.data = data
        self.engineered_features = None
        
    def create_features(self):
        """Create comprehensive feature set"""
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        df = self.data.copy()
        
        # 1. Basic aggregations
        df['Total_GDP'] = df.sum(axis=1)
        df['Mean_Sector_Value'] = df.mean(axis=1)
        df['Median_Sector_Value'] = df.median(axis=1)
        
        # 2. Economic diversity indices
        df['Economic_Diversity_HHI'] = self._calculate_hhi(df)
        df['Economic_Diversity_Shannon'] = self._calculate_shannon_entropy(df)
        df['Economic_Diversity_Gini'] = self._calculate_gini(df)
        
        # 3. Sectoral dominance
        df['Max_Sector_Share'] = df.iloc[:, :-6].max(axis=1) / df['Total_GDP']
        df['Top3_Sectors_Share'] = self._top_n_share(df, 3)
        df['Sector_Concentration'] = df.iloc[:, :-8].std(axis=1) / df['Mean_Sector_Value']
        
        # 4. Economic structure indicators
        df['Primary_Sector_Share'] = self._calculate_primary_share(df)
        df['Secondary_Sector_Share'] = self._calculate_secondary_share(df)
        df['Tertiary_Sector_Share'] = self._calculate_tertiary_share(df)
        df['Modernization_Index'] = (df['Secondary_Sector_Share'] + 
                                     df['Tertiary_Sector_Share'])
        
        # 5. Statistical measures
        df['Sector_Variance'] = df.iloc[:, :len(self.data.columns)].var(axis=1)
        df['Sector_Skewness'] = df.iloc[:, :len(self.data.columns)].skew(axis=1)
        df['Sector_Kurtosis'] = df.iloc[:, :len(self.data.columns)].kurtosis(axis=1)
        
        # 6. Relative measures (compared to national average)
        for col in self.data.columns:
            if 'Sector_' in col:
                nat_avg = df[col].mean()
                if nat_avg > 0:
                    df[f'{col}_Relative'] = df[col] / nat_avg
        
        self.engineered_features = df
        
        print(f"Created {len(df.columns) - len(self.data.columns)} new features")
        print(f"Total features: {len(df.columns)}")
        
        # Feature importance summary
        self._print_feature_summary()
        
        return df
    
    def _calculate_hhi(self, df):
        """Calculate Herfindahl-Hirschman Index"""
        sector_cols = [col for col in df.columns if 'Sector_' in col]
        hhi = []
        for _, row in df.iterrows():
            total = row[sector_cols].sum()
            if total > 0:
                shares = (row[sector_cols] / total) ** 2
                hhi.append(1 - shares.sum())  # 1 - HHI for diversity
            else:
                hhi.append(0)
        return hhi
    
    def _calculate_shannon_entropy(self, df):
        """Calculate Shannon entropy for diversity"""
        sector_cols = [col for col in df.columns if 'Sector_' in col]
        entropy = []
        for _, row in df.iterrows():
            total = row[sector_cols].sum()
            if total > 0:
                shares = row[sector_cols] / total
                shares = shares[shares > 0]  # Remove zeros
                H = -np.sum(shares * np.log(shares))
                entropy.append(H)
            else:
                entropy.append(0)
        return entropy
    
    def _calculate_gini(self, df):
        """Calculate Gini coefficient for inequality"""
        sector_cols = [col for col in df.columns if 'Sector_' in col]
        gini = []
        for _, row in df.iterrows():
            values = row[sector_cols].values
            values = values[values > 0]
            if len(values) > 0:
                sorted_values = np.sort(values)
                n = len(sorted_values)
                cumsum = np.cumsum(sorted_values)
                gini_val = (2 * np.sum((np.arange(1, n+1)) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n
                gini.append(gini_val)
            else:
                gini.append(0)
        return gini
    
    def _top_n_share(self, df, n=3):
        """Calculate share of top N sectors"""
        sector_cols = [col for col in df.columns if 'Sector_' in col]
        shares = []
        for _, row in df.iterrows():
            total = row[sector_cols].sum()
            if total > 0:
                top_n = row[sector_cols].nlargest(n).sum()
                shares.append(top_n / total)
            else:
                shares.append(0)
        return shares
    
    def _calculate_primary_share(self, df):
        """Calculate primary sector share (Agriculture, Mining)"""
        primary_sectors = ['Sector_A', 'Sector_B']
        available = [col for col in primary_sectors if col in df.columns]
        if available:
            return df[available].sum(axis=1) / df['Total_GDP']
        return pd.Series(0, index=df.index)
    
    def _calculate_secondary_share(self, df):
        """Calculate secondary sector share (Manufacturing, Construction)"""
        secondary_sectors = ['Sector_C', 'Sector_D', 'Sector_E', 'Sector_F']
        available = [col for col in secondary_sectors if col in df.columns]
        if available:
            return df[available].sum(axis=1) / df['Total_GDP']
        return pd.Series(0, index=df.index)
    
    def _calculate_tertiary_share(self, df):
        """Calculate tertiary sector share (Services)"""
        tertiary_sectors = ['Sector_G', 'Sector_H', 'Sector_I', 'Sector_J', 
                           'Sector_K', 'Sector_L', 'Sector_M', 'Sector_N']
        available = [col for col in tertiary_sectors if col in df.columns]
        if available:
            return df[available].sum(axis=1) / df['Total_GDP']
        return pd.Series(0, index=df.index)
    
    def _print_feature_summary(self):
        """Print summary of engineered features"""
        print("\nFeature Engineering Summary:")
        print("-" * 40)
        feature_groups = {
            'Diversity Measures': ['HHI', 'Shannon', 'Gini'],
            'Structural Indicators': ['Primary', 'Secondary', 'Tertiary', 'Modernization'],
            'Statistical Measures': ['Variance', 'Skewness', 'Kurtosis'],
            'Concentration Measures': ['Max_Sector', 'Top3', 'Concentration']
        }
        
        for group, keywords in feature_groups.items():
            features = [f for f in self.engineered_features.columns 
                       if any(k in f for k in keywords)]
            if features:
                print(f"{group}: {len(features)} features")

# ==========================================
# PART 3: CLUSTERING ANALYSIS
# ==========================================

class ProvincialClustering:
    """Advanced clustering analysis for provincial economies"""
    
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.scaled_data = None
        self.clustering_results = {}
        
    def prepare_data(self):
        """Prepare and scale data for clustering"""
        # Select numeric features
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.scaled_data = self.scaler.fit_transform(self.data[numeric_cols])
        return self.scaled_data
    
    def find_optimal_clusters(self, max_k=5):
        """Find optimal number of clusters using multiple metrics"""
        print("\n" + "="*50)
        print("FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("="*50)
        
        K_range = range(2, min(max_k + 1, len(self.data)))
        
        metrics = {
            'inertia': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_data)
            
            metrics['inertia'].append(kmeans.inertia_)
            metrics['silhouette'].append(silhouette_score(self.scaled_data, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(self.scaled_data, labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(self.scaled_data, labels))
        
        # Plot metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Elbow curve
        axes[0, 0].plot(K_range, metrics['inertia'], 'bo-')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method')
        axes[0, 0].grid(True)
        
        # Silhouette score
        axes[0, 1].plot(K_range, metrics['silhouette'], 'ro-')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Score (Higher is Better)')
        axes[0, 1].grid(True)
        
        # Davies-Bouldin score
        axes[1, 0].plot(K_range, metrics['davies_bouldin'], 'go-')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Davies-Bouldin Score')
        axes[1, 0].set_title('Davies-Bouldin Score (Lower is Better)')
        axes[1, 0].grid(True)
        
        # Calinski-Harabasz score
        axes[1, 1].plot(K_range, metrics['calinski_harabasz'], 'mo-')
        axes[1, 1].set_xlabel('Number of Clusters')
        axes[1, 1].set_ylabel('Calinski-Harabasz Score')
        axes[1, 1].set_title('Calinski-Harabasz Score (Higher is Better)')
        axes[1, 1].grid(True)
        
        plt.suptitle('Clustering Optimization Metrics', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'optimal_clusters_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Determine optimal k
        optimal_k = K_range[np.argmax(metrics['silhouette'])]
        
        print(f"Optimal number of clusters based on Silhouette Score: {optimal_k}")
        print("\nMetrics Summary:")
        for k_idx, k in enumerate(K_range):
            print(f"k={k}: Silhouette={metrics['silhouette'][k_idx]:.3f}, "
                  f"Davies-Bouldin={metrics['davies_bouldin'][k_idx]:.3f}")
        
        return optimal_k, metrics
    
    def perform_kmeans(self, n_clusters=3):
        """Perform K-means clustering"""
        print("\n" + "="*50)
        print("K-MEANS CLUSTERING")
        print("="*50)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = kmeans.fit_predict(self.scaled_data)
        
        self.clustering_results['kmeans'] = {
            'model': kmeans,
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'silhouette': silhouette_score(self.scaled_data, labels)
        }
        
        # Analyze clusters
        self._analyze_clusters(labels, 'K-means')
        
        return labels
    
    def perform_hierarchical(self, n_clusters=3):
        """Perform Hierarchical clustering"""
        print("\n" + "="*50)
        print("HIERARCHICAL CLUSTERING")
        print("="*50)
        
        # Create linkage matrix
        linkage_matrix = linkage(self.scaled_data, method='ward')
        
        # Get clusters
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
        
        self.clustering_results['hierarchical'] = {
            'linkage_matrix': linkage_matrix,
            'labels': labels,
            'silhouette': silhouette_score(self.scaled_data, labels)
        }
        
        # Create dendrogram
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, 
                  labels=self.data.index.tolist(),
                  leaf_rotation=45,
                  leaf_font_size=12)
        plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=14)
        plt.xlabel('Province')
        plt.ylabel('Distance')
        plt.axhline(y=linkage_matrix[-n_clusters+1, 2], color='r', linestyle='--', 
                   label=f'Cut for {n_clusters} clusters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, 'hierarchical_dendrogram.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Analyze clusters
        self._analyze_clusters(labels, 'Hierarchical')
        
        return labels
    
    # REMOVED: perform_dbscan() - Not used in core analysis
    # Density-based clustering not appropriate for n=7 provinces
    # Uncomment if needed for additional validation
    
    # REMOVED: perform_gmm() - Not used in core analysis
    # GMM adds probabilistic complexity without changing cluster assignments for our data
    # Uncomment if needed for additional validation
    
    def _analyze_clusters(self, labels, method_name):
        """Analyze and characterize clusters"""
        print(f"\n{method_name} Cluster Analysis:")
        print("-" * 40)
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise points
                continue
                
            provinces = self.data.index[labels == cluster_id].tolist()
            print(f"\nCluster {cluster_id}: {provinces}")
            
            # Calculate cluster characteristics
            cluster_data = self.data.iloc[labels == cluster_id]
            
            if 'Economic_Diversity_HHI' in cluster_data.columns:
                avg_diversity = cluster_data['Economic_Diversity_HHI'].mean()
                print(f"  Avg Economic Diversity: {avg_diversity:.3f}")
            
            if 'Modernization_Index' in cluster_data.columns:
                avg_modern = cluster_data['Modernization_Index'].mean()
                print(f"  Avg Modernization Index: {avg_modern:.3f}")
            
            if 'Total_GDP' in cluster_data.columns:
                avg_gdp = cluster_data['Total_GDP'].mean()
                print(f"  Avg Total GDP: {avg_gdp:,.0f}")
    
    def compare_methods(self):
        """Compare clustering methods"""
        print("\n" + "="*50)
        print("CLUSTERING METHOD COMPARISON")
        print("="*50)
        
        comparison_df = pd.DataFrame(index=self.data.index)
        
        for method, results in self.clustering_results.items():
            if 'labels' in results:
                comparison_df[method] = results['labels']
        
        # Calculate agreement matrix
        methods = list(comparison_df.columns)
        n_methods = len(methods)
        agreement_matrix = np.zeros((n_methods, n_methods))
        
        for i in range(n_methods):
            for j in range(n_methods):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    labels1 = comparison_df[methods[i]].values
                    labels2 = comparison_df[methods[j]].values
                    # Calculate adjusted rand index
                    from sklearn.metrics import adjusted_rand_score
                    agreement_matrix[i, j] = adjusted_rand_score(labels1, labels2)
        
        # Plot agreement heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(agreement_matrix, annot=True, fmt='.2f', 
                   xticklabels=methods, yticklabels=methods,
                   cmap='coolwarm', center=0.5, vmin=0, vmax=1)
        plt.title('Clustering Method Agreement (Adjusted Rand Index)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_agreement.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\nMethod Agreement Scores:")
        for i in range(n_methods):
            for j in range(i+1, n_methods):
                print(f"{methods[i]} vs {methods[j]}: {agreement_matrix[i, j]:.3f}")
        
        return comparison_df

# ==========================================
# PART 4: CLASSIFICATION ANALYSIS
# ==========================================

class ProvincialClassification:
    """Advanced classification for provincial development tiers"""
    
    def __init__(self, data):
        self.data = data
        self.X = None
        self.y = None
        self.classifiers = {}
        self.results = {}
        
    def create_labels(self, method='composite'):
        """Create development tier labels"""
        print("\n" + "="*50)
        print("CREATING DEVELOPMENT LABELS")
        print("="*50)
        
        if method == 'composite':
            # Create composite development index
            scores = pd.Series(index=self.data.index, dtype=float)
            
            # Factors for development score
            if 'Total_GDP' in self.data.columns:
                gdp_score = (self.data['Total_GDP'] - self.data['Total_GDP'].min()) / \
                           (self.data['Total_GDP'].max() - self.data['Total_GDP'].min() + 1e-10)
            else:
                gdp_score = pd.Series(0.5, index=self.data.index)
            
            if 'Economic_Diversity_HHI' in self.data.columns:
                diversity_score = self.data['Economic_Diversity_HHI']
            else:
                diversity_score = pd.Series(0.5, index=self.data.index)
            
            if 'Modernization_Index' in self.data.columns:
                modern_score = self.data['Modernization_Index']
            else:
                modern_score = pd.Series(0.5, index=self.data.index)
            
            # Weighted composite
            scores = (gdp_score * 0.4 + diversity_score * 0.3 + modern_score * 0.3)
            
            # Create labels based on tertiles
            tertiles = scores.quantile([0.33, 0.67])
            
            labels = pd.Series(index=self.data.index, dtype=str)
            labels[scores <= tertiles.iloc[0]] = 'Lagging'
            labels[(scores > tertiles.iloc[0]) & (scores <= tertiles.iloc[1])] = 'Emerging'
            labels[scores > tertiles.iloc[1]] = 'Advanced'
            
        else:
            # Alternative: use clustering results
            if 'kmeans' in self.clustering_results:
                cluster_labels = self.clustering_results['kmeans']['labels']
                label_map = {0: 'Type_A', 1: 'Type_B', 2: 'Type_C'}
                labels = pd.Series([label_map.get(l, 'Unknown') for l in cluster_labels],
                                 index=self.data.index)
        
        self.y = labels
        
        print("\nDevelopment Tier Distribution:")
        print(labels.value_counts())
        
        for tier in labels.unique():
            provinces = labels[labels == tier].index.tolist()
            print(f"\n{tier}: {provinces}")
        
        return labels
    
    def prepare_features(self, feature_selection=True):
        """Prepare features for classification"""
        # Select numeric features
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.X = self.data[numeric_cols]
        
        if feature_selection and len(self.X.columns) > 10:
            # Feature selection using Random Forest importance
            print("\nPerforming feature selection...")
            
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_encoded = le.fit_transform(self.y)
            
            # Use Random Forest for feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(self.X, y_encoded)
            
            # Get feature importance
            importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Select top features
            n_features = min(10, len(self.X.columns) // 2)
            top_features = importance.head(n_features)['feature'].tolist()
            
            print(f"Selected {n_features} top features")
            print("Top 5 features:")
            for idx, row in importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
            
            self.X = self.X[top_features]
        
        return self.X
    
    def train_classifiers(self):
        """Train multiple classifiers"""
        print("\n" + "="*50)
        print("TRAINING CLASSIFIERS")
        print("="*50)
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Define classifiers - SIMPLIFIED to 2 core algorithms
        classifiers = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_split=2,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=50,
                max_depth=2,
                learning_rate=0.1,
                random_state=42
            )
            # REMOVED: XGBoost - overfits on n=7 (achieved only 14.3% accuracy)
        }
        
        # Train and evaluate each classifier
        for name, clf in classifiers.items():
            print(f"\nTraining {name}...")
            
            # Use Leave-One-Out cross-validation for small sample
            loo = LeaveOneOut()
            scores = cross_val_score(clf, X_scaled, y_encoded, cv=loo)
            
            # Train on full dataset
            clf.fit(X_scaled, y_encoded)
            
            # Store results
            self.classifiers[name] = {
                'model': clf,
                'scaler': scaler,
                'label_encoder': le,
                'cv_scores': scores,
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std()
            }
            
            print(f"  LOO CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
            
            # Feature importance for tree-based models
            if hasattr(clf, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': self.X.columns,
                    'importance': clf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"  Top 3 important features:")
                for idx, row in importance.head(3).iterrows():
                    print(f"    {row['feature']}: {row['importance']:.3f}")
        
        return self.classifiers
    
    def create_ensemble(self):
        """Create ensemble classifier"""
        print("\n" + "="*50)
        print("ENSEMBLE CLASSIFIER")
        print("="*50)
        
        # Create voting classifier
        estimators = []
        for name, clf_dict in self.classifiers.items():
            estimators.append((name, clf_dict['model']))
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        # Encode labels
        le = self.classifiers['Random Forest']['label_encoder']
        y_encoded = le.transform(self.y)
        
        # Scale features
        scaler = self.classifiers['Random Forest']['scaler']
        X_scaled = scaler.transform(self.X)
        
        # Cross-validate ensemble
        loo = LeaveOneOut()
        scores = cross_val_score(ensemble, X_scaled, y_encoded, cv=loo)
        
        # Train ensemble
        ensemble.fit(X_scaled, y_encoded)
        
        self.classifiers['Ensemble'] = {
            'model': ensemble,
            'scaler': scaler,
            'label_encoder': le,
            'cv_scores': scores,
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std()
        }
        
        print(f"Ensemble LOO CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
        
        return ensemble

# ==========================================
# PART 5: VISUALIZATION AND REPORTING
# ==========================================

class VisualizationReporter:
    """Create comprehensive visualizations and reports"""
    
    def __init__(self, data, clustering_results, classification_results):
        self.data = data
        self.clustering_results = clustering_results
        self.classification_results = classification_results
        
    def create_pca_visualization(self):
        """Create PCA visualization of clusters"""
        print("\nCreating PCA visualization...")
        
        # Prepare data
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data[numeric_cols])
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(scaled_data)
        
        # Create plots for each clustering method
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Changed from 2x2 to 1x2 for 2 methods
        methods = ['kmeans', 'hierarchical']  # SIMPLIFIED - only core methods

        for idx, method in enumerate(methods):
            ax = axes[idx]
            
            if method in self.clustering_results and 'labels' in self.clustering_results[method]:
                labels = self.clustering_results[method]['labels']
                
                # Plot
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                   c=labels, cmap='viridis', 
                                   s=200, alpha=0.7, edgecolors='black')
                
                # Add province names
                for i, province in enumerate(self.data.index):
                    ax.annotate(province.replace('Province ', 'P'),
                              (X_pca[i, 0], X_pca[i, 1]),
                              ha='center', va='center', fontsize=9, fontweight='bold')
                
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                ax.set_title(f'{method.upper()} Clustering')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'{method.upper()}\nNot Available',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle('Provincial Clusters - PCA Visualization', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'pca_clusters_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_tsne_visualization(self):
        """Create t-SNE visualization"""
        print("Creating t-SNE visualization...")
        
        # Prepare data
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data[numeric_cols])
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=3)
        X_tsne = tsne.fit_transform(scaled_data)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        if 'kmeans' in self.clustering_results:
            labels = self.clustering_results['kmeans']['labels']
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                c=labels, cmap='Set1', 
                                s=300, alpha=0.7, edgecolors='black')
            
            # Add province names
            for i, province in enumerate(self.data.index):
                plt.annotate(province, (X_tsne[i, 0], X_tsne[i, 1]),
                           ha='center', va='bottom', fontsize=10)
            
            plt.colorbar(scatter, label='Cluster')
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('Provincial Economic Structure - t-SNE Visualization')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_visualization.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_report(self):
        """Create comprehensive analysis report"""
        print("\nGenerating comprehensive report...")
        
        # Create summary dashboard
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Cluster sizes
        ax1 = plt.subplot(3, 3, 1)
        if 'kmeans' in self.clustering_results:
            labels = self.clustering_results['kmeans']['labels']
            unique, counts = np.unique(labels, return_counts=True)
            ax1.bar(unique, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax1.set_xlabel('Cluster')
            ax1.set_ylabel('Number of Provinces')
            ax1.set_title('K-means Cluster Distribution')
            ax1.set_xticks(unique)
        
        # 2. Development tier distribution
        ax2 = plt.subplot(3, 3, 2)
        if self.classification_results and hasattr(self.classification_results, 'y'):
            tier_counts = self.classification_results.y.value_counts()
            colors = {'Advanced': 'green', 'Emerging': 'yellow', 'Lagging': 'red'}
            bar_colors = [colors.get(x, 'gray') for x in tier_counts.index]
            ax2.bar(tier_counts.index, tier_counts.values, color=bar_colors)
            ax2.set_xlabel('Development Tier')
            ax2.set_ylabel('Count')
            ax2.set_title('Provincial Development Classification')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Economic diversity comparison
        ax3 = plt.subplot(3, 3, 3)
        if 'Economic_Diversity_HHI' in self.data.columns:
            provinces = self.data.index
            diversity = self.data['Economic_Diversity_HHI'].values
            ax3.barh(range(len(provinces)), diversity)
            ax3.set_yticks(range(len(provinces)))
            ax3.set_yticklabels([p.replace('Province ', 'P') for p in provinces])
            ax3.set_xlabel('Economic Diversity (HHI)')
            ax3.set_title('Provincial Economic Diversity')
            ax3.grid(True, alpha=0.3)
        
        # 4. Feature importance
        ax4 = plt.subplot(3, 3, 4)
        if self.classification_results and 'Random Forest' in self.classification_results.classifiers:
            rf = self.classification_results.classifiers['Random Forest']['model']
            if hasattr(rf, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': self.classification_results.X.columns,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=True).tail(8)
                
                ax4.barh(range(len(importance)), importance['importance'].values)
                ax4.set_yticks(range(len(importance)))
                ax4.set_yticklabels([f[:20] for f in importance['feature'].values])
                ax4.set_xlabel('Importance')
                ax4.set_title('Top Feature Importance (RF)')
        
        # 5. Clustering agreement matrix
        ax5 = plt.subplot(3, 3, 5)
        methods_comparison = []
        methods_names = []
        for method in ['kmeans', 'hierarchical']:  # SIMPLIFIED - only core methods
            if method in self.clustering_results and 'labels' in self.clustering_results[method]:
                methods_comparison.append(self.clustering_results[method]['labels'])
                methods_names.append(method)
        
        if len(methods_comparison) >= 2:
            from sklearn.metrics import adjusted_rand_score
            n_methods = len(methods_comparison)
            agreement = np.zeros((n_methods, n_methods))
            
            for i in range(n_methods):
                for j in range(n_methods):
                    agreement[i, j] = adjusted_rand_score(methods_comparison[i], 
                                                         methods_comparison[j])
            
            im = ax5.imshow(agreement, cmap='coolwarm', vmin=0, vmax=1)
            ax5.set_xticks(range(n_methods))
            ax5.set_yticks(range(n_methods))
            ax5.set_xticklabels(methods_names, rotation=45)
            ax5.set_yticklabels(methods_names)
            ax5.set_title('Clustering Agreement (ARI)')
            
            # Add values
            for i in range(n_methods):
                for j in range(n_methods):
                    ax5.text(j, i, f'{agreement[i, j]:.2f}',
                           ha='center', va='center')
        
        # 6. Modernization vs GDP
        ax6 = plt.subplot(3, 3, 6)
        if 'Modernization_Index' in self.data.columns and 'Total_GDP' in self.data.columns:
            ax6.scatter(self.data['Modernization_Index'], 
                       self.data['Total_GDP'],
                       s=200, alpha=0.7)
            
            for i, province in enumerate(self.data.index):
                ax6.annotate(province.replace('Province ', 'P'),
                           (self.data['Modernization_Index'].iloc[i],
                            self.data['Total_GDP'].iloc[i]),
                           ha='center', va='bottom')
            
            ax6.set_xlabel('Modernization Index')
            ax6.set_ylabel('Total GDP')
            ax6.set_title('Modernization vs Economic Size')
            ax6.grid(True, alpha=0.3)
        
        # 7. Silhouette scores comparison
        ax7 = plt.subplot(3, 3, 7)
        silhouette_scores = []
        method_names = []
        for method, results in self.clustering_results.items():
            if 'silhouette' in results and results['silhouette'] > -1:
                silhouette_scores.append(results['silhouette'])
                method_names.append(method)
        
        if silhouette_scores:
            ax7.bar(method_names, silhouette_scores, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax7.set_ylabel('Silhouette Score')
            ax7.set_title('Clustering Quality Comparison')
            ax7.set_ylim([0, 1])
            ax7.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(silhouette_scores):
                ax7.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 8. Classification accuracy comparison
        ax8 = plt.subplot(3, 3, 8)
        if self.classification_results and self.classification_results.classifiers:
            accuracies = []
            classifier_names = []
            
            for name, clf_dict in self.classification_results.classifiers.items():
                accuracies.append(clf_dict['mean_accuracy'])
                classifier_names.append(name)
            
            ax8.bar(classifier_names, accuracies, 
                   color=['#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#9467bd'])
            ax8.set_ylabel('LOO CV Accuracy')
            ax8.set_title('Classifier Performance Comparison')
            ax8.set_ylim([0, 1])
            plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax8.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(accuracies):
                ax8.text(i, v + 0.01, f'{v:.2%}', ha='center')
        
        # 9. Sectoral composition heatmap
        ax9 = plt.subplot(3, 3, 9)
        sector_cols = [col for col in self.data.columns if col.startswith('Sector_')][:8]
        if sector_cols:
            sector_data = self.data[sector_cols].T
            sector_data.index = [s.replace('Sector_', '') for s in sector_data.index]
            sector_data.columns = [p.replace('Province ', 'P') for p in sector_data.columns]
            
            # Normalize by column (province)
            sector_norm = sector_data.div(sector_data.sum(axis=0), axis=1)
            
            im = ax9.imshow(sector_norm.values, cmap='YlOrRd', aspect='auto')
            ax9.set_xticks(range(len(sector_data.columns)))
            ax9.set_yticks(range(len(sector_data.index)))
            ax9.set_xticklabels(sector_data.columns)
            ax9.set_yticklabels(sector_data.index)
            ax9.set_title('Sectoral Composition (Normalized)')
            ax9.set_xlabel('Province')
            ax9.set_ylabel('Sector')
        
        plt.suptitle('Nepal Provincial Economic ML Analysis - Comprehensive Dashboard',
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'comprehensive_dashboard.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Dashboard created successfully!")
    
    def save_results(self):
        """Save all results to files"""
        print("\nSaving results...")
        
        # Create results dataframe
        results_df = self.data.copy()
        
        # Add clustering results
        for method, results in self.clustering_results.items():
            if 'labels' in results:
                results_df[f'cluster_{method}'] = results['labels']
        
        # Add classification results
        if self.classification_results and hasattr(self.classification_results, 'y'):
            results_df['development_tier'] = self.classification_results.y
        
        # Save to CSV
        results_df.to_csv(os.path.join(OUTPUT_DIR, 'provincial_analysis_results.csv'))
        
        # Save summary statistics
        summary = {
            'n_provinces': len(self.data),
            'n_features': len(self.data.columns),
            'clustering_methods': list(self.clustering_results.keys()),
            'classification_methods': list(self.classification_results.classifiers.keys()) 
                                    if self.classification_results else [],
            'best_clustering_silhouette': max(
                [r.get('silhouette', -1) for r in self.clustering_results.values()]
            ),
            'best_classification_accuracy': max(
                [c['mean_accuracy'] for c in self.classification_results.classifiers.values()]
            ) if self.classification_results else 0
        }
        
        with open(os.path.join(OUTPUT_DIR, 'analysis_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print("Results saved successfully!")
        print(f"  - Results CSV: {os.path.join(OUTPUT_DIR, 'provincial_analysis_results.csv')}")
        print(f"  - Summary JSON: {os.path.join(OUTPUT_DIR, 'analysis_summary.json')}")
        print(f"  - Visualizations: {OUTPUT_DIR}/*.png files")

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

def main():
    """Execute complete ML pipeline"""
    print("="*60)
    print("NEPAL PROVINCIAL ECONOMIC ML ANALYSIS")
    print("Production Implementation with Actual Data")
    print("="*60)
    
    # Step 1: Extract Data
    print("\n" + "="*50)
    print("STEP 1: DATA EXTRACTION")
    print("="*50)
    
    # Use relative path to data file
    data_path = os.path.join('data', 'National-Accounts.xlsx')

    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        print("Please ensure National-Accounts.xlsx is in the 'data' directory")
        return

    extractor = ProvincialDataExtractor(data_path)
    raw_data = extractor.load_data()
    
    if raw_data is None or raw_data.empty:
        print("ERROR: Could not extract data. Please check the file structure.")
        return
    
    # Step 2: Feature Engineering
    print("\n" + "="*50)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*50)
    
    engineer = FeatureEngineer(raw_data)
    engineered_data = engineer.create_features()
    
    # Step 3: Clustering Analysis
    print("\n" + "="*50)
    print("STEP 3: CLUSTERING ANALYSIS")
    print("="*50)
    
    clustering = ProvincialClustering(engineered_data)
    clustering.prepare_data()
    
    # Find optimal clusters
    optimal_k, metrics = clustering.find_optimal_clusters(max_k=4)
    
    # Perform multiple clustering methods
    # CORE CLUSTERING ALGORITHMS (Simplified from 4 to 2)
    clustering.perform_kmeans(n_clusters=optimal_k)
    clustering.perform_hierarchical(n_clusters=optimal_k)
    # REMOVED: clustering.perform_gmm() and clustering.perform_dbscan() - not in core analysis
    
    # Compare methods
    comparison = clustering.compare_methods()
    
    # Step 4: Classification Analysis
    print("\n" + "="*50)
    print("STEP 4: CLASSIFICATION ANALYSIS")
    print("="*50)
    
    classification = ProvincialClassification(engineered_data)
    classification.clustering_results = clustering.clustering_results  # Pass clustering results
    
    # Create labels
    classification.create_labels(method='composite')
    
    # Prepare features
    classification.prepare_features(feature_selection=True)
    
    # Train classifiers
    classification.train_classifiers()
    
    # Create ensemble
    classification.create_ensemble()
    
    # Step 5: Visualization and Reporting
    print("\n" + "="*50)
    print("STEP 5: VISUALIZATION AND REPORTING")
    print("="*50)
    
    reporter = VisualizationReporter(engineered_data, 
                                    clustering.clustering_results,
                                    classification)
    
    reporter.create_pca_visualization()
    reporter.create_tsne_visualization()
    reporter.create_comprehensive_report()
    reporter.save_results()
    
    # Final Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    
    print("\nKey Findings:")
    print("-" * 40)
    
    # Best clustering method
    best_clustering = max(clustering.clustering_results.items(), 
                         key=lambda x: x[1].get('silhouette', -1))
    print(f"Best Clustering Method: {best_clustering[0].upper()}")
    print(f"  Silhouette Score: {best_clustering[1]['silhouette']:.3f}")
    
    # Best classifier
    if classification.classifiers:
        best_classifier = max(classification.classifiers.items(),
                            key=lambda x: x[1]['mean_accuracy'])
        print(f"\nBest Classifier: {best_classifier[0]}")
        print(f"  LOO CV Accuracy: {best_classifier[1]['mean_accuracy']:.1%}")
    
    print("\nGenerated Files:")
    print("  1. optimal_clusters_analysis.png")
    print("  2. hierarchical_dendrogram.png")
    print("  3. clustering_agreement.png")
    print("  4. pca_clusters_comparison.png")
    print("  5. tsne_visualization.png")
    print("  6. comprehensive_dashboard.png")
    print("  7. provincial_analysis_results.csv")
    print("  8. analysis_summary.json")

    # # Step 6: Generate IEEE Publication-Quality Images
    # print("\n" + "="*50)
    # print("STEP 6: GENERATING IEEE PAPER IMAGES")
    # print("="*50)

    # try:
    #     import subprocess
    #     result = subprocess.run(['python', 'create_paper_images.py'],
    #                           capture_output=True, text=True, check=True)
    #     print(result.stdout)
    #     print("\n[SUCCESS] IEEE-compliant images generated for academic paper!")
    # except subprocess.CalledProcessError as e:
    #     print(f"[WARNING] Could not generate IEEE images: {e}")
    #     print("You can manually run: python create_paper_images.py")
    # except FileNotFoundError:
    #     print("[WARNING] create_paper_images.py not found")
    #     print("IEEE images will not be generated automatically")
    
    return clustering, classification, reporter

if __name__ == "__main__":
    clustering, classification, reporter = main()
