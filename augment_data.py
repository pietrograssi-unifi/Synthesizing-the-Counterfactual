"""
Description:
    This script implements a generative machine learning pipeline to augment 
    underpowered longitudinal clinical data. It synthesizes counterfactual 
    trajectories for treated (Palliative Care) and control (Standard Care) cohorts.

Models Implemented:
    1. Causal TTVAE (Transformer-based Time-Series VAE with Causal Masking)
    2. TVAE / CTGAN (via the SDV library) as state-of-the-art benchmarks.
"""

import os
import sys
import warnings
import traceback
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Optional dependencies for benchmark generative models
try:
    from ctgan import CTGAN, TVAE
    CTGAN_AVAILABLE = True
except ImportError:
    CTGAN_AVAILABLE = False

# Suppress non-critical warnings for clean execution logging
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

# Training Hyperparameters
AUGMENTATION_FACTOR = 1.0
EPOCHS_TTVAE = 150
EPOCHS_GAN = 150
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
LATENT_DIM = 64

# Data Schema & Identifiers
ID_COL = 'survivor_id'
TIME_COL = 'rel_time'
SPLIT_COL = 'treat_group'  # 1 = Palliative Care, 0 = Standard Care
PADDING_VAL = -999.0       # Standardized token for post-attrition padding

# Structural Skeleton: Immutable baseline covariates preserved from empirical data
SKELETON_STATIC = [
    'wave_death', 'death_year', 'is_female', 'country', 'control_eligible', 
    'cause_death', 'edu_level', 'living_area_cat', 
    'dep_anchor', 'wealth_anchor'
]

# Structural Skeleton: Time-varying variables that initialize the temporal sequence
SKELETON_START = ['age', 'wave', 'int_year']

# Static variables synthesized by the generative model
STATIC_GEN = ['gender_imp', 'hc125_num', 'is_treated']

# Dynamic (longitudinal) variables synthesized by the generative model
DYNAMIC_VARS = [
    'rel_time', 
    'income_imp', 'wealth_imp', 'sphus_imp', 'adl_imp', 'maxgrip_imp', 
    'eurod_imp', 'dep_score', 'wealth_log', 'adl_raw', 'adl_score', 'maxgrip',
    'ins_sat', 'health_sat', 'satisfaction_health',
    'has_cancer', 'has_neuro', 'has_organ',
    'ph006d1', 'ph006d4', 'ph006d6', 'ph006d10', 'ph006d12', 'ph006d16', 'ph006d21'
]

# Deterministic evolutionary rules applied during wide-to-long reconstruction
DETERMINISTIC_RULES = {
    'age': 2,       # Age increases strictly by 2 years per survey wave
    'wave': 1,      # Wave increments strictly by 1
    'int_year': 2   # Interview year increments strictly by 2
}

# Hardware Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 2. NEURAL NETWORK ARCHITECTURE (Causal TTVAE)
# ==============================================================================

class CausalTTVAE_Model(nn.Module):
    """
    Transformer-based Time-Series Variational Autoencoder (TTVAE) with Causal Masking.
    
    OBJECTIVE: 
    Learn the complex joint probability distribution of longitudinal panel data 
    to generate high-fidelity synthetic trajectories.
    
    KEY COMPONENTS:
    1. Transformer Encoder: Compresses the temporal sequence into latent representations.
    2. Causal Masking: Enforces autoregressive constraints, preventing forward-looking 
       information leakage (at time 't', the model can only attend to steps 0 to t).
    3. Latent Space (VAE): Introduces stochasticity for generative sampling.
    4. Transformer Decoder: Reconstructs the temporal sequence from the latent space.
    """
    def __init__(self, input_dim: int, seq_len: int, d_model: int = 128, 
                 nhead: int = 4, num_layers: int = 2, latent_dim: int = 64):
        super(CausalTTVAE_Model, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 1. Input Projection: Maps raw features to the internal transformer dimensionality
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. Learnable Positional Encoding: Injects sequence order information
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # 3. Transformer Encoder: Captures complex temporal dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, 
            batch_first=True, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Latent Space Parameterization (VAE)
        self.fc_mu = nn.Linear(d_model * seq_len, latent_dim)
        self.fc_logvar = nn.Linear(d_model * seq_len, latent_dim)
        self.fc_z_to_seq = nn.Linear(latent_dim, d_model * seq_len)
        
        # 5. Transformer Decoder: Reconstructs the temporal sequence
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, 
            batch_first=True, dropout=0.1
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # 6. Output Head: Projects internal dimensionality back to the original feature space
        self.output_head = nn.Linear(d_model, input_dim)

    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        """
        Generates an upper-triangular mask to enforce autoregressive causality.
        Values of '-inf' effectively nullify future attention weights after softmax.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(DEVICE)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Encoder architecture."""
        batch_size = x.size(0)
        
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        mask = self._generate_causal_mask(x.size(1))
        x = self.transformer_encoder(x, mask=mask)
        
        x = x.reshape(batch_size, -1) 
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to enable backpropagation through stochastic sampling.
        Z = mu + sigma * epsilon (where epsilon ~ N(0,1)).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Decoder architecture."""
        batch_size = z.size(0)
        
        x = self.fc_z_to_seq(z)
        x = x.reshape(batch_size, self.seq_len, self.d_model)
        
        mask = self._generate_causal_mask(self.seq_len)
        x = self.transformer_decoder(x, mask=mask)
        return self.output_head(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Executes the full forward pass: Encoder -> Latent Sampling -> Decoder."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class TTVAE_Wrapper:
    """
    Wrapper class for end-to-end data processing, 3D tensor conversion, 
    model training, and generative sampling.
    """
    def __init__(self, epochs: int = 200, batch_size: int = 128, latent_dim: int = 64):
        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.encoders = {}
        self.model = None
        
    def fit(self, df_wide: pd.DataFrame, n_visits: int, static_cols: List[str], prob_cols: List[str]):
        """
        Prepares the empirical data and trains the TTVAE model.
        Converts the wide-format DataFrame into a 3D Tensor of shape:
        (N_Patients, N_TimeSteps, N_Features).
        """
        self.static_cols = static_cols
        self.prob_cols = prob_cols
        
        # 1. Categorical Encoding
        data = df_wide.copy()
        for col in data.columns:
            if data[col].dtype == 'object':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.encoders[col] = le
        self.feature_names = list(data.columns)
        
        # 2. MinMax Scaling
        data_for_fit = data.replace(PADDING_VAL, 0)
        self.scaler.fit(data_for_fit)
        data_matrix = self.scaler.transform(data.replace(PADDING_VAL, 0))
        
        static_indices = [self.feature_names.index(c) for c in static_cols if c in self.feature_names]
        
        # 3. Sequential Tensor Construction (Merging static and dynamic features)
        sequences = []
        for i in range(len(data)):
            row_raw = data.iloc[i] 
            row_scaled = data_matrix[i]
            static_vals = row_scaled[static_indices]
            
            seq_steps = []
            for v in range(n_visits):
                step_vals = []
                
                # Padding Mask Mechanism: 1 for valid observations, 0 for post-attrition padding
                check_col = f"{prob_cols[0]}_v{v}"
                if check_col not in row_raw: check_col = f"dep_score_v{v}"
                
                is_padding = False
                if check_col in row_raw:
                     try:
                         if float(row_raw[check_col]) <= PADDING_VAL + 1: is_padding = True
                     except: pass
                
                mask_val = 0.0 if is_padding else 1.0
                
                for pc in prob_cols:
                    col_name = f"{pc}_v{v}"
                    if col_name in self.feature_names:
                        idx = self.feature_names.index(col_name)
                        val = row_scaled[idx]
                        if is_padding: val = 0.0 
                        step_vals.append(val)
                    else: 
                        step_vals.append(0.0)
                
                full_step = np.concatenate([static_vals, step_vals, [mask_val]])
                seq_steps.append(full_step)
            sequences.append(seq_steps)
            
        self.X_train = np.array(sequences, dtype=np.float32)
        
        # 4. Model Initialization & Training Loop
        input_dim = self.X_train.shape[2]
        self.model = CausalTTVAE_Model(input_dim=input_dim, seq_len=n_visits, latent_dim=self.latent_dim).to(DEVICE)
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss(reduction='none') 
        
        tensor_x = torch.FloatTensor(self.X_train).to(DEVICE)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        print(f"   [INITIALIZATION] TTVAE Model instantiated. Input Dimension: {input_dim}")
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0]
                optimizer.zero_grad()
                recon_x, mu, logvar = self.model(x)
                
                # Reconstruction Loss (MSE)
                loss_mse = loss_fn(recon_x, x)
                
                # Loss Weighting: Heavier penalty (5.0) on the padding mask prediction 
                # to strictly enforce structural boundaries (e.g., patient mortality).
                weights = torch.ones_like(loss_mse)
                weights[:, :, -1] = 5.0 
                recon_loss = (loss_mse * weights).mean()
                
                # Kullback-Leibler Divergence (KLD) for latent space regularization
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
                
                loss = recon_loss + 0.002 * kld_loss 
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch+1) % 50 == 0:
                print(f"      Epoch {epoch+1}/{self.epochs} | Loss: {total_loss / len(loader):.4f}")

    def sample(self, n_samples: int) -> pd.DataFrame:
        """
        Synthesizes new longitudinal trajectories by sampling from the latent space
        and applying inverse scaling/encoding.
        """
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(DEVICE)
            recon_seq = self.model.decode(z).cpu().numpy()
            
        static_indices = [self.feature_names.index(c) for c in self.static_cols if c in self.feature_names]
        n_static = len(static_indices)
        n_dyn = len(self.prob_cols)
        
        final_matrix = np.zeros((n_samples, len(self.feature_names)))
        col_map = {name: i for i, name in enumerate(self.feature_names)}
        padding_masks = np.zeros((n_samples, recon_seq.shape[1]))
        
        # Matrix Reconstruction
        for i in range(n_samples):
            static_vec = recon_seq[i, 0, :n_static]
            for s_i, col_idx in enumerate(static_indices):
                final_matrix[i, col_idx] = static_vec[s_i]
            
            for v in range(recon_seq.shape[1]):
                step = recon_seq[i, v]
                dyn_vals = step[n_static : n_static+n_dyn]
                padding_masks[i, v] = step[-1] 
                
                for d_i, col in enumerate(self.prob_cols):
                    full_col = f"{col}_v{v}"
                    if full_col in col_map:
                        final_matrix[i, col_map[full_col]] = dyn_vals[d_i]
                        
        # Inverse Scaling
        data_inv = self.scaler.inverse_transform(final_matrix)
        df_syn = pd.DataFrame(data_inv, columns=self.feature_names)
        
        # Post-Processing: Categorical Decoding and Integer Constraints
        for col in df_syn.columns:
            if col in self.encoders:
                le = self.encoders[col]
                df_syn[col] = df_syn[col].round().clip(0, len(le.classes_)-1).astype(int)
                df_syn[col] = le.inverse_transform(df_syn[col])
            else:
                if any(x in col for x in ['has_', 'is_', '_imp', 'n_waves']):
                     df_syn[col] = df_syn[col].round()
        
        # Enforce Padding Mask based on network predictions (< 0.5 implies exit)
        for i in range(n_samples):
            for v in range(recon_seq.shape[1]):
                if padding_masks[i, v] < 0.5:
                    for pc in self.prob_cols:
                        col_name = f"{pc}_v{v}"
                        if col_name in df_syn.columns:
                            df_syn.at[i, col_name] = PADDING_VAL
        return df_syn


# ==============================================================================
# 3. DATA PROCESSING HELPERS
# ==============================================================================

def prepare_wide_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Transforms longitudinal data (long format) into wide format.
    Essential for standard neural architectures that require fixed-length 
    input vectors rather than sparse panel structures.
    """
    print("   [PRE-PROCESSING] Executing longitudinal-to-wide transformation...")
    
    # Establish relative event-time centered on the mortality event
    if 'rel_time' not in df.columns and 'wave' in df.columns and 'wave_death' in df.columns:
        df['rel_time'] = df['wave'] - df['wave_death']
        
    df_sorted = df.sort_values(by=[ID_COL, 'wave'])
    grouped = df_sorted.groupby(ID_COL)
    
    wide_rows = []
    max_waves_obs = df[ID_COL].value_counts().max()
    
    for survivor_id, group in grouped:
        row = {}
        row['n_waves_real'] = len(group)
        first = group.iloc[0]
        
        # A. Static Variables
        for col in SKELETON_STATIC + STATIC_GEN + [SPLIT_COL]:
            if col in df.columns: row[col] = first[col]
            
        # B. Initial conditions for deterministic evolution rules
        for col in SKELETON_START: 
            if col in first: row[f"{col}_start"] = first[col]
            
        # C. Dynamic Variables (Flattening the temporal sequence)
        for i in range(max_waves_obs):
            suffix = f"_v{i}"
            
            if i < len(group):
                for col in DYNAMIC_VARS: 
                    if col in group.columns: row[col + suffix] = group.iloc[i][col]
            else:
                # Post-attrition periods are filled with the standardized padding token
                for col in DYNAMIC_VARS: row[col + suffix] = PADDING_VAL
                
        wide_rows.append(row)
        
    return pd.DataFrame(wide_rows), max_waves_obs


def reconstruct_long_data(df_wide: pd.DataFrame, max_waves: int) -> pd.DataFrame:
    """
    Inverse transformation: Wide to Long format.
    Implements dynamic sequence truncation based on predicted padding tokens,
    effectively simulating structural study exit (mortality/attrition).
    """
    print("   [POST-PROCESSING] Reconstructing longitudinal format from generated sequences...")
    long_rows = []
    
    for idx, row in df_wide.iterrows():
        syn_id = f"SYN_{idx}"
        
        # Quality Control: Discard phantom trajectories where t=0 is predicted as padded
        if row.get(f"{DYNAMIC_VARS[0]}_v0", 0) <= PADDING_VAL + 1: continue

        static_vals = {col: row[col] for col in (SKELETON_STATIC + STATIC_GEN) if col in row}
        if SPLIT_COL in row: static_vals[SPLIT_COL] = row[SPLIT_COL]
        
        curr_vals = {}
        for col in DETERMINISTIC_RULES.keys():
            if f"{col}_start" in row: curr_vals[col] = row[f"{col}_start"]
        
        n_waves_pred = int(row.get('n_waves_real', max_waves))
        n_waves_pred = max(1, min(max_waves, n_waves_pred))
        
        for i in range(n_waves_pred):
            # Conditional sequence truncation simulating temporal attrition
            if row.get(f"{DYNAMIC_VARS[0]}_v{i}", 0) <= PADDING_VAL + 1: break
            
            long_row = {}
            long_row[ID_COL] = syn_id
            long_row.update(static_vals)
            long_row.update(curr_vals)
            
            for col in DYNAMIC_VARS:
                long_row[col] = row.get(f"{col}_v{i}", 0)
            
            # Mathematical deduction of relative time to ensure structural integrity
            if 'wave' in curr_vals and 'wave_death' in static_vals:
                long_row['rel_time'] = curr_vals['wave'] - static_vals['wave_death']
            
            long_rows.append(long_row)
            
            # Enforce deterministic evolutionary rules (e.g., aging)
            for col, step in DETERMINISTIC_RULES.items():
                if col in curr_vals: curr_vals[col] += step
                
    return pd.DataFrame(long_rows)


def robust_sample(model, n_required: int) -> pd.DataFrame:
    """
    Robust sampling mechanism with oversampling to actively filter out 
    collapsed or pure-padding synthetic trajectories, ensuring exact support target.
    """
    valid_samples = []
    attempts = 0
    
    while len(valid_samples) < n_required and attempts < 10:
        n_missing = n_required - len(valid_samples)
        batch = model.sample(int(n_missing * 1.5) + 10)
        
        check_col = f"{DYNAMIC_VARS[0]}_v0"
        if check_col not in batch.columns: check_col = "dep_score_v0"
        
        if check_col in batch.columns:
            batch = batch[batch[check_col] > (PADDING_VAL + 10)]
            
        if len(batch) > 0: valid_samples.append(batch)
        attempts += 1
        
    return pd.concat(valid_samples).iloc[:n_required] if valid_samples else pd.DataFrame()


def preprocess_causal_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain-specific empirical boundaries prior to generative modeling 
    to enforce logical data structures.
    """
    print("   [PRE-PROCESSING] Applying empirical boundaries and causal constraints...")
    df_smooth = df.copy()
    if 'dep_score' in df_smooth.columns: 
        df_smooth['dep_score'] = df_smooth['dep_score'].clip(0, 12)
    return df_smooth


def compute_anchors_simple(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts baseline (t=-1) structural anchors for the Skeleton Injection mechanism."""
    if 'rel_time' not in df.columns:
        if 'wave_death' in df.columns: df['rel_time'] = df['wave'] - df['wave_death']
        else: return df 
        
    anchors = df[df['rel_time'] == -1][['survivor_id', 'dep_score', 'wealth_log']]
    anchors = anchors.rename(columns={'dep_score': 'dep_anchor', 'wealth_log': 'wealth_anchor'})
    
    if 'dep_anchor' not in df.columns:
        df = df.merge(anchors, on='survivor_id', how='left')
        
    if 'dep_anchor' in df.columns:
        df['dep_anchor'] = df['dep_anchor'].fillna(df['dep_score'].mean())
        df['wealth_anchor'] = df['wealth_anchor'].fillna(df['wealth_log'].mean())
        
    return df


# ==============================================================================
# 4. MAIN GENERATION PIPELINE
# ==============================================================================

def train_and_generate_split(df_base: pd.DataFrame, model_type: str = 'TTVAE') -> pd.DataFrame:
    print(f"\n=== Executing Causal Synthesis Pipeline | Architecture: {model_type} ===")
    
    df_base = compute_anchors_simple(df_base)
    df_wide, max_waves = prepare_wide_data(df_base)
    syn_dfs = []
    
    n_treated_orig = len(df_wide[df_wide[SPLIT_COL] == 1])
    target_n = int(n_treated_orig * AUGMENTATION_FACTOR)
    
    for group_val in [0, 1]: 
        df_group = df_wide[df_wide[SPLIT_COL] == group_val].copy()
        
        if len(df_group) < 10: 
            print(f"   [WARNING] Insufficient empirical support for Cohort {group_val}. Skipping.")
            continue
        
        print(f"   [COHORT {group_val}] Empirical Baseline: {len(df_group)} units -> Target Synthesis: {target_n} units")
        train_data = df_group.drop(columns=[SPLIT_COL], errors='ignore')
        
        # --- SKELETON EXTRACTION ---
        skel_static_cols = [c for c in (SKELETON_STATIC + STATIC_GEN) if c in train_data.columns]
        skel_start_cols = [f"{c}_start" for c in SKELETON_START if f"{c}_start" in train_data.columns]
        
        real_skeleton = train_data[skel_static_cols + skel_start_cols].values
        features_static = skel_static_cols + skel_start_cols
        
        syn_data = None
        
        # --- TRAINING PHASE ---
        if model_type == 'TTVAE':
            wrapper = TTVAE_Wrapper(epochs=EPOCHS_TTVAE, latent_dim=LATENT_DIM)
            wrapper.fit(train_data, max_waves, features_static, DYNAMIC_VARS)
            syn_data = robust_sample(wrapper, target_n)
            
        elif CTGAN_AVAILABLE:
            try:
                discrete_cols = [c for c in train_data.columns if train_data[c].dtype == 'object' or train_data[c].nunique() < 20]
                if model_type == 'CTGAN': model = CTGAN(epochs=EPOCHS_GAN, verbose=True)
                else: model = TVAE(epochs=EPOCHS_GAN)
                model.fit(train_data, discrete_columns=discrete_cols)
                syn_data = robust_sample(model, target_n)
            except Exception as e: 
                print(f"   [ERROR] Architectural failure in {model_type}: {e}")

        # --- EMPIRICAL SKELETON INJECTION ---
        # Anchors synthetic dynamic trajectories to exact real-world baseline coordinates
        if syn_data is not None and len(real_skeleton) > 0:
            indices = np.random.choice(len(real_skeleton), size=len(syn_data), replace=True)
            sampled_skel = real_skeleton[indices]
            
            for i, col in enumerate(skel_static_cols + skel_start_cols):
                if col in syn_data.columns: 
                    syn_data[col] = sampled_skel[:, i]
            
            syn_data[SPLIT_COL] = group_val
            syn_dfs.append(syn_data)
            
    if not syn_dfs: return pd.DataFrame()
    
    df_syn_wide = pd.concat(syn_dfs, ignore_index=True)
    return reconstruct_long_data(df_syn_wide, max_waves)


def main():
    print("=== LONGITUDINAL AUGMENTATION PIPELINE (FINAL SPECIFICATION) ===")
    
    if not os.path.exists('BaseDataset.csv'): 
        print("[CRITICAL] Input matrix 'BaseDataset.csv' not found. Halting execution."); return
        
    df_base = pd.read_csv('BaseDataset.csv')
    df_base['is_synthetic'] = 0
    
    # 1. Initial Empirical Pre-processing
    if 'dep_score' in df_base.columns: df_base['dep_score'] = df_base['dep_score'].clip(0, 12)
    if 'wealth_log' in df_base.columns: df_base['wealth_log'] = df_base['wealth_log'].clip(0, 20)
    
    # --- DOMAIN CLAMPING SETUP ---
    # Calculates empirical boundaries from the observational sample to restrict 
    # generative artifacts and guarantee clinical/logical realism.
    numeric_cols = df_base.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_clamp = [c for c in numeric_cols if c not in [ID_COL, 'is_synthetic', 'treat_group']]
    
    real_ranges = {}
    for col in cols_to_clamp:
        valid_vals = df_base[df_base[col] > -990][col] 
        if not valid_vals.empty:
            real_ranges[col] = (valid_vals.min(), valid_vals.max())
        else:
            real_ranges[col] = (0, 1) 
            
    print(f"   [SETUP] Structural constraints dynamically calculated for {len(real_ranges)} variables.")

    df_prep = preprocess_causal_trends(df_base)
    
    models_to_run = ['TTVAE']
    if CTGAN_AVAILABLE: models_to_run += ['TVAE', 'CTGAN']
    
    for m in models_to_run:
        try:
            syn_df = train_and_generate_split(df_prep, model_type=m)
            
            if not syn_df.empty:
                syn_df['is_synthetic'] = 1
                
                # --- POST-PROCESSING: DOMAIN CLAMPING APPLICATION ---
                print(f"   [POST-PROCESSING] Enforcing empirical boundaries for {m} output...")
                
                for col in cols_to_clamp:
                    if col in syn_df.columns:
                        min_v, max_v = real_ranges[col]
                        
                        # 1. Resolve padding artifacts
                        syn_df.loc[syn_df[col] <= -990, col] = min_v
                        
                        # 2. Enforce empirical distribution boundaries
                        syn_df[col] = syn_df[col].clip(lower=min_v, upper=max_v)
                        
                        # 3. Enforce integer constraints on clinical/discrete measures
                        if col in ['dep_score', 'hc125_num', 'eurod_imp'] or 'has_' in col:
                             syn_df[col] = syn_df[col].round()

                # Structural Assembly
                common_cols = df_base.columns.intersection(syn_df.columns)
                final_df = pd.concat([df_base[common_cols], syn_df[common_cols]], ignore_index=True)
                
                print(f"   [ASSEMBLY] Hybrid Panel Constructed: {len(df_base)} Empirical + {len(syn_df)} Synthetic units.")
            else:
                final_df = df_base.copy()
                print(f"   [WARNING] Generation failed for {m}. Returning baseline empirical data.")

            # Export Pipeline Output
            fname = f"Augmented_v1new_{m}.csv"
            final_df.to_csv(fname, index=False)
            print(f"\n[EXPORT] Augmented dataset successfully saved to: {fname}\n")
            
        except Exception as e:
            print(f"   [CRITICAL] Execution failed for architecture {m}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
