# file: utils/metrics.py
def edit_distance(ref_words, hyp_words):
    """
    Compute Levenshtein edit distance between two lists of words.
    Returns the number of substitutions, deletions, and insertions.
    """
    N = len(ref_words)
    M = len(hyp_words)
    # Initialize DP table
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    for i in range(1, N+1):
        dp[i][0] = i
    for j in range(1, M+1):
        dp[0][j] = j
    # Compute edit distance DP
    for i in range(1, N+1):
        for j in range(1, M+1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,        # deletion
                           dp[i][j-1] + 1,        # insertion
                           dp[i-1][j-1] + cost)   # substitution
    return dp[N][M]

def word_error_rate(ref_text, hyp_text):
    """
    Compute Word Error Rate (WER) given reference and hypothesis text strings.
    """
    # Normalize text: lowercase, remove punctuation for evaluation
    ref_text = ref_text.strip().lower()
    hyp_text = hyp_text.strip().lower()
    # Split into words
    ref_words = ref_text.split()
    hyp_words = hyp_text.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    # Compute edit distance
    dist = edit_distance(ref_words, hyp_words)
    wer = dist / len(ref_words)
    return wer

import math
import numpy as np

def mel_cepstral_distance(mel_spec_ref, mel_spec_pred):
    """
    Compute Mel Cepstral Distortion (MCD) between reference and predicted mel-spectrograms.
    Uses 13-dimensional MFCC (excluding C0) with dynamic time warping alignment.
    mel_spec_ref/pred: numpy arrays of shape (T, n_mels) in dB (log mel).
    Returns MCD in dB.
    """
    # Convert to numpy if tensor
    if torch.is_tensor(mel_spec_ref):
        mel_spec_ref = mel_spec_ref.cpu().numpy()
    if torch.is_tensor(mel_spec_pred):
        mel_spec_pred = mel_spec_pred.cpu().numpy()
    # Compute DCT (type-II) to get MFCC (we'll use 13 coefficients)
    # We can use np.fft.dct for simplicity (which is type-II by default)
    def compute_mfcc(log_mel):
        # log_mel: shape (T, n_mels)
        # Compute DCT along mel axis
        mfcc = np.zeros((log_mel.shape[0], 13))
        for t in range(log_mel.shape[0]):
            # DCT-II manually for first 13 coefficients
            for m in range(13):
                # ortho-normalization factor could be applied, but we'll ignore for simplicity
                vals = [log_mel[t, k] * math.cos(math.pi * m * (2*k+1) / (2 * log_mel.shape[1])) for k in range(log_mel.shape[1])]
                mfcc[t, m] = sum(vals)
        return mfcc
    mfcc_ref = compute_mfcc(mel_spec_ref)
    mfcc_pred = compute_mfcc(mel_spec_pred)
    # Exclude 0th coefficient (energy)
    mfcc_ref = mfcc_ref[:, 1:13]
    mfcc_pred = mfcc_pred[:, 1:13]
    # DTW alignment on MFCC sequences
    N, M = mfcc_ref.shape[0], mfcc_pred.shape[0]
    # distance matrix
    dist = np.zeros((N+1, M+1))
    dist.fill(np.inf)
    dist[0, 0] = 0
    for i in range(1, N+1):
        for j in range(1, M+1):
            # Euclidean distance between MFCC vectors at frame i-1 and j-1
            cost = np.linalg.norm(mfcc_ref[i-1] - mfcc_pred[j-1])
            # accumulate minimal path cost
            dist[i, j] = cost + min(dist[i-1, j], dist[i, j-1], dist[i-1, j-1])
    total_cost = dist[N, M]
    path_length = 0
    # Backtrace to find path length
    i, j = N, M
    while i > 0 or j > 0:
        path_length += 1
        # move in direction of minimal cumulative cost
        diag = dist[i-1, j-1] if i>0 and j>0 else np.inf
        up = dist[i-1, j] if i>0 else np.inf
        left = dist[i, j-1] if j>0 else np.inf
        if diag <= up and diag <= left:
            i -= 1
            j -= 1
        elif up < left:
            i -= 1
        else:
            j -= 1
    # Average cost per frame
    avg_cost = total_cost / path_length
    # MCD formula in dB
    mcd = (10.0 / math.log(10)) * math.sqrt(2) * avg_cost
    return mcd
