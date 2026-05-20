import librosa
import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline
import os

# =========================================================================
# --- AGENT CODE INJECTED HERE ---
def apply_dsp_filter(y, sr):
    """
    Identifies and suppresses heavy laughter in an audio signal to improve diarization.
    This filter works by:
    1. Extracting frame-level features indicative of laughter (energy, zero-crossing rate, spectral flatness).
    2. Combining these features into a single "laughter score".
    3. Thresholding this score to create a binary mask identifying potential laughter segments.
    4. Applying temporal post-processing (median filtering, morphological closing) to the mask
       to remove spurious detections and create contiguous segments.
    5. Smoothing the mask and applying it to the original audio to suppress laughter.
    """
    # NOTE: It is assumed that numpy, librosa, and scipy.ndimage are available in the execution environment.
    import numpy as np
    import librosa
    from scipy.ndimage import median_filter, binary_closing

    # 1. Define analysis parameters
    n_fft = 2048
    hop_length = 512

    # 2. Feature Extraction
    # Extract features sensitive to the acoustic properties of laughter (noisy, high energy bursts).
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length).flatten()
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length).flatten()
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length).flatten()

    # Ensure all feature vectors have the same length
    min_len = min(len(rms), len(zcr), len(flatness))
    rms, zcr, flatness = rms[:min_len], zcr[:min_len], flatness[:min_len]

    # 3. Feature Normalization and Scoring
    def normalize(x, eps=1e-8):
        # Min-Max normalization to a 0-1 range
        max_val, min_val = np.max(x), np.min(x)
        if (max_val - min_val) < eps:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    # Normalize features to be on a comparable scale
    rms_norm = normalize(rms)
    zcr_norm = normalize(zcr)
    flatness_norm = normalize(flatness)

    # Combine features into a laughter score. Laughter is often noisy (high ZCR, high flatness)
    # and has high energy (high RMS). We weigh ZCR and flatness more heavily.
    laughter_score = (0.2 * rms_norm) + (0.4 * zcr_norm) + (0.4 * flatness_norm)

    # 4. Thresholding for Initial Mask
    # Use a percentile-based threshold, robust to outliers. Identify top 20% of scores as potential laughter.
    threshold = np.percentile(laughter_score, 80)
    
    # Add an energy-based floor to prevent silence or faint noise from being classified as laughter.
    energy_threshold = np.percentile(rms, 25)

    # Create initial mask: 1.0 for laughter, 0.0 for non-laughter
    initial_mask = ((laughter_score > threshold) & (rms > energy_threshold)).astype(float)

    # 5. Mask Post-Processing for Temporal Consistency
    # Median filtering removes short, isolated 'salt-and-pepper' errors.
    median_win_len = 7  # frames (~80 ms)
    filtered_mask = median_filter(initial_mask, size=median_win_len)
    
    # Morphological closing fills small gaps within detected laughter segments.
    closing_win_len = 15  # frames (~170 ms)
    structure = np.ones(closing_win_len)
    closed_mask = binary_closing(filtered_mask, structure=structure).astype(float)

    # 6. Final Mask Generation for Audio Application
    # The `closed_mask` has 1 for laughter. We want to suppress this.
    # The final multiplier mask should be 0 for laughter and 1 for speech.
    # We apply a smoothing window to create soft transitions to avoid audible clicks.
    smoothing_win_len = hop_length // 2  # ~10ms smoothing window
    smoothing_win = np.hanning(smoothing_win_len)
    smoothing_win /= np.sum(smoothing_win)
    
    # Smooth the laughter mask (where 1=laughter) before inverting.
    smoothed_laughter_mask = np.convolve(closed_mask, smoothing_win, mode='same')
    
    # Invert to get a "speech preservation" mask.
    speech_preservation_mask = 1.0 - smoothed_laughter_mask
    
    # 7. Upsample Mask and Apply to Audio
    # Repeat each frame's mask value for `hop_length` samples.
    expanded_mask = np.repeat(speech_preservation_mask, hop_length)
    
    # Ensure the expanded mask's length matches the original audio exactly.
    len_y = len(y)
    if len(expanded_mask) > len_y:
        expanded_mask = expanded_mask[:len_y]
    else:
        padding = np.full(len_y - len(expanded_mask), expanded_mask[-1])
        expanded_mask = np.concatenate((expanded_mask, padding))

    # Apply the final mask to the audio signal.
    y_masked = y * expanded_mask

    # Ensure the final output is a strictly 1D numpy array.
    return y_masked.flatten()
# =========================================================================

def process_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Isolate DSP logic in the agent's function
    y_masked = apply_dsp_filter(y, sr)
    
    temp_wav = "temp_masked_audio.wav"
    sf.write(temp_wav, y_masked, sr)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=os.environ.get("HF_TOKEN")
    )
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    diarization = pipeline(temp_wav)
    
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
        
    return diarization
