import torch
from torch_audiomentations import Compose, PitchShift, Gain, PolarityInversion

# =========================================================================
# --- AGENT CODE INJECTED HERE ---
def get_agent_configuration():
    """
    Returns the hyperparameter dictionary and augmentation pipeline.
    """
    # STRATEGY: Initial conservative baseline. Slight gain adjustments.
    hyperparameters = {
        "learning_rate": 1e-4,
        "batch_size": 8,
        "vad_onset": 0.5,     
        "vad_offset": 0.5,
        "min_speech_duration": 0.1
    }

    augmentation_pipeline = Compose(
        transforms=[
            Gain(min_gain_in_db=-3.0, max_gain_in_db=3.0, p=0.5)
        ]
    )
    
    return hyperparameters, augmentation_pipeline
# =========================================================================