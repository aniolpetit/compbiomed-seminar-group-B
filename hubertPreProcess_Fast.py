import math, numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt, resample_poly

LEADS = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']

# Design the IIR once, reuse everywhere
def design_bandpass_sos(fs, low=0.05, high=47, order=4):
    wp = [low / (fs / 2), high / (fs / 2)]
    return butter(order, wp, btype='band', output='sos')

def preprocess_block(block, sos, orig_fs, target_fs=100, duration=5):
    """
    block: np.ndarray shape (12, n_samples)
    returns: (12, 500) float32
    """
    # 1. Band-pass all leads at once
    block = sosfiltfilt(sos, block, axis=-1)

    # 2. Resample together
    g = math.gcd(orig_fs, target_fs)
    block = resample_poly(block, target_fs // g, orig_fs // g, axis=-1)

    # 3. Crop / zero-pad to exactly duration s
    wanted = target_fs * duration
    if block.shape[-1] >= wanted:
        block = block[..., :wanted]
    else:
        pad = wanted - block.shape[-1]
        block = np.pad(block, ((0, 0), (0, pad)))

    # 4. Normalise each lead to [-1, 1]
    max_abs = np.max(np.abs(block), axis=-1, keepdims=True)
    max_abs[max_abs == 0] = 1          # avoid /0
    block = block / max_abs

    return block.astype(np.float32)
    
def preprocess_dataframe_fast(df: pd.DataFrame,
                              orig_fs: int = 250,
                              target_fs: int = 100,
                              duration: int = 5,
                              n_jobs: int = -1) -> pd.DataFrame:
    sos = design_bandpass_sos(orig_fs)
    
    def row_to_block(row):
        # stack 12 leads → (12, n_samples)
        return np.vstack(row[LEADS].values)
    
    # -------- parallel map --------
    processed = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
        delayed(preprocess_block)(
            row_to_block(row), sos, orig_fs, target_fs, duration
        )
        for _, row in df.iterrows()
    )

    # -------- put back into a new DataFrame --------
    out = df.copy()
    for i, lead in enumerate(LEADS):
        out[lead] = [blk[i] for blk in processed]
    
    return out

ecg_df = pd.read_pickle("ecg_dataset.pkl")
ecg_df_pp = preprocess_dataframe_fast(
    ecg_df,           
    orig_fs=250,
    target_fs=100,
    duration=5,
    n_jobs=-1         
)

from pathlib import Path
out_file = Path.cwd() / "ecg_dataset_preprocessed_fast.pkl"
ecg_df_pp.to_pickle(out_file, protocol=4)