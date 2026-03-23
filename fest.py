"""
==============================================================================
SYNTHETIC DATA EVALUATION FRAMEWORK (FEST)
==============================================================================

Description:
    This module implements a comprehensive comparative assessment framework 
    to benchmark synthetic data generation architectures (CTGAN, TVAE, TTVAE) 
    against ground-truth observational panel data.

Methodology:
    The framework operationalizes a "Dual-Axis" evaluation strategy:
    
    1. Statistical Fidelity & Causal Utility:
       - Marginal Distributions: Two-sample Kolmogorov-Smirnov (KS) Test.
       - Structural Integrity: Frobenius Norm of Correlation Matrix Differences.
       - First-Order Moments: Mean Absolute Percentage Error (MAPE).
       - Causal Dynamics: Pre-treatment Parallel Trends Stability (critical 
         for Difference-in-Differences estimation) and Anchor Consistency.
       
    2. Privacy Preservation & Disclosure Risk:
       - Overfitting/Memorization: Distance to Closest Record (DCR).
       - Indistinguishability: Adversarial Accuracy via Random Forest classifiers
         (Propensity to discriminate synthetic from empirical records).

Author: Pietro Grassi
Publication Context: Q1 Journal Submission (JRSS A)
Date: March 2026
==============================================================================
"""

import os
import warnings
from typing import Dict, List, Any, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

# Suppress non-critical warnings to ensure clean execution logging
warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION A: BASE COMPUTATIONAL ARCHITECTURE
# ==============================================================================

class BaseCalculator:
    """
    Abstract base class for metric computation.
    
    Responsibility:
    Ensures structural isomorphism between the empirical (Real) and generated 
    (Synthetic) feature spaces. It handles schema intersections and applies 
    deterministic central-tendency imputation solely to enable the mathematical 
    computation of distance-based metrics (e.g., KS, DCR), without altering 
    the underlying generative model evaluation.
    """
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                 real_name: str = "Real", synthetic_name: str = "Synthetic"):
        
        # Retain raw synthetic data to access auxiliary generative artifacts 
        # (e.g., Structural Anchors) that may not reside in the empirical subset.
        self.syn_raw = synthetic_data.copy()
        
        # Enforce numeric casting for structural anchor variables
        for anchor in ['dep_anchor', 'wealth_anchor', 'dep_score', 'wealth_log']:
            if anchor in self.syn_raw.columns:
                self.syn_raw[anchor] = pd.to_numeric(self.syn_raw[anchor], errors='coerce')
        
        self.syn_raw.fillna(0, inplace=True)
        
        # 1. Feature Space Alignment (Intersection Logic)
        # Metrics are strictly computed on the orthogonal subset of shared covariates.
        self.common_cols = real_data.columns.intersection(synthetic_data.columns)
        
        self.real = real_data[self.common_cols].copy()
        self.syn = synthetic_data[self.common_cols].copy()
        self.r_name = real_name
        self.s_name = synthetic_name
        
        # 2. Variable Typology Detection
        self.num_cols = self.real.select_dtypes(include=np.number).columns
        self.cat_cols = self.real.select_dtypes(exclude=np.number).columns
        
        # 3. Preprocessing for Metric Stability
        if len(self.num_cols) > 0:
            self.real[self.num_cols] = self.real[self.num_cols].fillna(self.real[self.num_cols].mean())
            self.syn[self.num_cols] = self.syn[self.num_cols].fillna(self.syn[self.num_cols].mean())
        
        if len(self.cat_cols) > 0:
            for col in self.cat_cols:
                mode_val = self.real[col].mode()[0]
                self.real[col] = self.real[col].fillna(mode_val)
                self.syn[col] = self.syn[col].fillna(mode_val)


class MetricManager:
    """Abstract orchestrator for executing metric computation batches."""
    def __init__(self): 
        self.metrics = []
        
    def add_metric(self, m: Union[BaseCalculator, List[BaseCalculator]]): 
        if isinstance(m, list): 
            self.metrics.extend(m)
        else: 
            self.metrics.append(m)
            
    def evaluate_all(self) -> Dict[str, float]:
        results = {}
        for metric in self.metrics: 
            results.update(metric.calculate())
        return results


# ==============================================================================
# SECTION B: FIDELITY METRICS (CAUSAL UTILITY)
# ==============================================================================

class BasicStatsCalculator(BaseCalculator):
    """
    Evaluates the preservation of first-order moments (Central Tendency).
    Metric: Mean Absolute Percentage Error (MAPE).
    """
    def calculate(self) -> Dict[str, float]:
        print(f"   [FIDELITY] Computing First-Order Moment Preservation (MAPE)...")
        if len(self.num_cols) == 0: return {}
        
        real_mean = self.real[self.num_cols].mean()
        syn_mean = self.syn[self.num_cols].mean()
        
        # Epsilon scalar added to the denominator to prevent zero-division artifacts
        mape = np.mean(np.abs((real_mean - syn_mean) / (real_mean + 1e-6)))
        return {f"{self.s_name}_Mean_MAPE": mape}


class KSCalculator(BaseCalculator):
    """
    Kolmogorov-Smirnov (KS) Test.
    Quantifies marginal distributional fidelity by measuring the maximum distance 
    between the Empirical Cumulative Distribution Functions (ECDF).
    Output: Average KS Complement (1.0 - statistic). 1.0 implies perfect overlap.
    """
    def calculate(self) -> Dict[str, float]:
        print(f"   [FIDELITY] Computing Marginal Distributional Fidelity (KS Test)...")
        if len(self.num_cols) == 0: return {}
        
        ks_scores = []
        for col in self.num_cols:
            stat, _ = ks_2samp(self.real[col], self.syn[col])
            ks_scores.append(1 - stat) 
            
        return {f"{self.s_name}_KS_Score_Avg": np.mean(ks_scores)}


class CorrelationCalculator(BaseCalculator):
    """
    Structural Correlation Analysis.
    Computes the Frobenius Norm of the difference between empirical and synthetic 
    correlation matrices (||R_emp - R_syn||_F) to assess multivariate dependency preservation.
    """
    def calculate(self) -> Dict[str, float]:
        print(f"   [FIDELITY] Computing Multivariate Structural Covariance (Frobenius Norm)...")
        
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        r_enc = self.real.copy()
        s_enc = self.syn.copy()
        
        if len(self.cat_cols) > 0:
            full_cat = pd.concat([r_enc[self.cat_cols].astype(str), s_enc[self.cat_cols].astype(str)])
            enc.fit(full_cat)
            r_enc[self.cat_cols] = enc.transform(r_enc[self.cat_cols].astype(str))
            s_enc[self.cat_cols] = enc.transform(s_enc[self.cat_cols].astype(str))

        corr_r = r_enc.corr()
        corr_s = s_enc.corr()
        
        diff_norm = np.linalg.norm(corr_r.fillna(0) - corr_s.fillna(0))
        return {f"{self.s_name}_Correlation_Diff": diff_norm}
    

class CausalFidelityCalculator(BaseCalculator):
    """
    Longitudinal Causal Fidelity Assessment.
    
    Evaluates the structural suitability of synthetic data for quasi-experimental 
    designs (Matched Difference-in-Differences).
    
    Metrics:
    1. Anchor Correlation: Verifies strict adherence to injected baseline skeletons.
    2. Parallel Trends Stability: Quantifies pre-treatment divergence (t-3 to t-1) 
       between treatment arms to detect algorithmically induced structural bias.
    """
    def calculate(self) -> Dict[str, float]:
        print(f"   [FIDELITY] Computing Causal Trajectory Metrics (Parallel Trends & Anchors)...")
        results = {}

        df_eval = self.syn_raw
        
        # 1. Structural Anchor Consistency Check
        for anchor_col in ['dep_anchor', 'wealth_anchor']:
            target_col = 'dep_score' if 'dep' in anchor_col else 'wealth_log'
            
            if anchor_col in df_eval.columns and target_col in df_eval.columns:
                
                # Isolate the baseline period (t = -1)
                if 'rel_time' in df_eval.columns:
                    base_slice = df_eval[df_eval['rel_time'] == -1]
                else:
                    base_slice = df_eval
                
                if not base_slice.empty:
                    corr = base_slice[target_col].corr(base_slice[anchor_col])
                    results[f"{self.s_name}_{target_col}_Anchor_Corr"] = corr
        
        # 2. Parallel Trends Stability (Pre-Treatment Slope Falsification)
        req_cols = ['rel_time', 'treat_group', 'dep_score']
        if all(c in df_eval.columns for c in req_cols):
            trends = df_eval.groupby(['rel_time', 'treat_group'])['dep_score'].mean().unstack()
            
            if 0 in trends.columns and 1 in trends.columns:
                trends['diff'] = trends[1] - trends[0]
                if -3 in trends.index and -1 in trends.index:
                    slope = abs(trends.loc[-1, 'diff'] - trends.loc[-3, 'diff'])
                    results[f"{self.s_name}_PreTrend_Stability_Slope"] = slope 
        
        return results


# ==============================================================================
# SECTION C: PRIVACY METRICS (DISCLOSURE RISK)
# ==============================================================================

class DCRCalculator(BaseCalculator):
    """
    Distance to Closest Record (DCR).
    Computes the Euclidean distance from each synthetic unit to its nearest 
    empirical neighbor. A DCR approaching zero indicates severe network overfitting 
    and high memorization risk.
    """
    def calculate(self) -> Dict[str, float]:
        print(f"   [PRIVACY] Computing Distance to Closest Record (Memorization Risk)...")
        if len(self.num_cols) == 0: return {}
        
        scaler = MinMaxScaler()
        r_scaled = scaler.fit_transform(self.real[self.num_cols])
        s_scaled = scaler.transform(self.syn[self.num_cols])
        
        # Stochastic subsampling for computational tractability (N > 3000)
        if len(s_scaled) > 3000:
            idx = np.random.choice(len(s_scaled), 3000, replace=False)
            s_scaled = s_scaled[idx]
        
        nbrs = NearestNeighbors(n_neighbors=1).fit(r_scaled)
        distances, _ = nbrs.kneighbors(s_scaled)
        
        return {f"{self.s_name}_DCR_Mean": np.mean(distances)}


class AdversarialAccuracyCalculator(BaseCalculator):
    """
    Adversarial Accuracy (AA).
    
    Simulates a linkage attack via a Random Forest discriminator trained to 
    classify records as empirical or synthetic. 
    An AA ~ 0.50 signifies theoretical indistinguishability; AA > 0.90 indicates 
    structural divergence or severe disclosure risk.
    """
    def calculate(self) -> Dict[str, float]:
        print(f"   [PRIVACY] Computing Adversarial Accuracy (Discriminator Susceptibility)...")
        X_real = self.real.copy()
        X_syn = self.syn.copy()
        X_real['label'] = 0
        X_syn['label'] = 1
        
        combined = pd.concat([X_real, X_syn], axis=0).sample(frac=1.0, random_state=42)
        y = combined['label']
        X = combined.drop('label', axis=1)
        
        for col in X.select_dtypes(include='object').columns:
            X[col] = pd.factorize(X[col])[0]
        X = X.fillna(0)
        
        if len(X) > 10000:
            X = X.iloc[:10000]
            y = y.iloc[:10000]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        clf.fit(X_train, y_train)
        
        return {f"{self.s_name}_Adversarial_Accuracy": accuracy_score(y_test, clf.predict(X_test))}


# ==============================================================================
# SECTION D: ANALYTICAL VISUALIZATION
# ==============================================================================

def plot_feature_distribution(real_df: pd.DataFrame, models_dict: Dict[str, pd.DataFrame], 
                              target_col: str, output_name: str = "Distribution_Comparison.png"):
    """Generates a comparative Kernel Density Estimation (KDE) plot."""
    if target_col not in real_df.columns: return

    plt.figure(figsize=(10, 6))
    sns.kdeplot(real_df[target_col], label='Empirical Cohort (Real)', fill=True, color='black', alpha=0.1, linewidth=2)
    
    colors = {'CTGAN': 'blue', 'TVAE': 'green', 'TTVAE': 'red'}
    for name, df in models_dict.items():
        if target_col in df.columns:
            sns.kdeplot(df[target_col], label=f'{name} (Synthetic)', 
                        color=colors.get(name, 'gray'), linestyle='--')
            
    plt.title(f"Marginal Distributional Consistency: EURO-D Scale", fontsize=14)
    plt.xlabel("EURO-D Depression Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"   [VISUALIZATION] Density distribution exported to '{output_name}'")


# ==============================================================================
# SECTION E: EVALUATION ORCHESTRATION PIPELINE
# ==============================================================================

def main():
    print("====================================================================")
    print("   FEST FRAMEWORK: SYNTHETIC DATA CAUSAL EVALUATION MODULE")
    print("====================================================================")
    
    # 1. Ingest Empirical Ground Truth
    if not os.path.exists('BaseDataset.csv'):
        print("[CRITICAL] Empirical baseline matrix 'BaseDataset.csv' not found. Halting.")
        return
    
    real_df = pd.read_csv('BaseDataset.csv')
    print(f"[INGESTION] Empirical Ground Truth Loaded. Dimensions: {real_df.shape}")

    # 2. Ingest Synthetic Candidates
    candidate_files = {
        'CTGAN': 'Augmented_v1new_CTGAN.csv',
        'TVAE': 'Augmented_v1new_TVAE.csv',
        'TTVAE': 'Augmented_v1new_TTVAE.csv'
    }
    
    models_data = {}
    for model_name, file_path in candidate_files.items():
        if os.path.exists(file_path):
            print(f"   -> Located Synthetic Counterfactuals: {model_name}")
            models_data[model_name] = pd.read_csv(file_path)

    if not models_data:
        print("[WARNING] No synthetic iterations detected. Execute generation pipeline prior to evaluation.")
        return

    # 3. Evaluation Loop Execution
    final_results = []

    for model_name, syn_df in models_data.items():
        print(f"\n--- Initiating Evaluation for Architecture: {model_name} ---")
        
        # Fidelity (Causal Utility) Assessment
        utility_manager = MetricManager()
        utility_manager.add_metric([
            BasicStatsCalculator(real_df, syn_df, synthetic_name=model_name),
            KSCalculator(real_df, syn_df, synthetic_name=model_name),
            CorrelationCalculator(real_df, syn_df, synthetic_name=model_name),
            CausalFidelityCalculator(real_df, syn_df, synthetic_name=model_name)
        ])
        u_res = utility_manager.evaluate_all()
        
        # Privacy (Disclosure Risk) Assessment
        privacy_manager = MetricManager()
        privacy_manager.add_metric([
            DCRCalculator(real_df, syn_df, synthetic_name=model_name),
            AdversarialAccuracyCalculator(real_df, syn_df, synthetic_name=model_name)
        ])
        p_res = privacy_manager.evaluate_all()
        
        # Metrics Consolidation
        full_metrics = {**u_res, **p_res}
        
        record = {
            'Model': model_name,
            'KS Score (Fidelity)': full_metrics.get(f"{model_name}_KS_Score_Avg", 0),
            'MAPE (Mean Error)': full_metrics.get(f"{model_name}_Mean_MAPE", 0),
            'Corr. Diff (Structure)': full_metrics.get(f"{model_name}_Correlation_Diff", 999),
            'DCR (Privacy)': full_metrics.get(f"{model_name}_DCR_Mean", 0),
            'Adv. Accuracy': full_metrics.get(f"{model_name}_Adversarial_Accuracy", 0.5),
            'Pre-Trend Slope': full_metrics.get(f"{model_name}_PreTrend_Stability_Slope", 999),
            'Dep Anchor Corr': full_metrics.get(f"{model_name}_dep_score_Anchor_Corr", 0),      
            'Wealth Anchor Corr': full_metrics.get(f"{model_name}_wealth_log_Anchor_Corr", 0) 
        }
        final_results.append(record)

    # 4. Report Generation
    results_df = pd.DataFrame(final_results)
    
    print("\n====================================================================")
    print("                 FINAL COMPARATIVE EVALUATION REPORT                  ")
    print("====================================================================")
    print(results_df.round(4).to_string(index=False))
    
    results_df.to_csv("FEST_Evaluation_Results.csv", index=False)
    
    # 5. Architecture Optimization Selection
    if not results_df.empty:
        winner = results_df.loc[results_df['KS Score (Fidelity)'].idxmax()]
        print(f"\n[EVALUATION COMPLETE] Optimal Generative Architecture: {winner['Model']}")
        print(f"   Rationale: Maximized marginal distributional fidelity (KS = {winner['KS Score (Fidelity)']:.4f})")

    # 6. Output Visualization
    target_col = 'eurod_imp' 
    if target_col not in real_df.columns:
        numerics = real_df.select_dtypes(include=np.number).columns
        if len(numerics) > 0:
            target_col = numerics[0]
            
    plot_feature_distribution(real_df, models_data, target_col)

if __name__ == "__main__":
    main()
