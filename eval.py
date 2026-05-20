import sys
import warnings
import os
from dotenv import load_dotenv
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
# Load the keys from the .env file
load_dotenv()

from extractor import process_audio 

# Suppress Pyannote/Librosa warnings for a clean stdout
warnings.filterwarnings("ignore")

def main():
    truth_rttm = "POD_711.rttm"
    audio_file = "POD_711.wav"
    
    try:
        truth_annotations = list(load_rttm(truth_rttm).values())[0]
        pred_annotation = process_audio(audio_file)
        
        metric = DiarizationErrorRate()
        der = metric(truth_annotations, pred_annotation, detailed=True)
        
        total_speech = der['total']
        missed = der['missed detection'] / total_speech
        false_alarm = der['false alarm'] / total_speech
        confusion = der['confusion'] / total_speech
        total_der = der['diarization error rate']

        # Constraint: If Missed Speech > 10% (approx 4 seconds), fail the run
        if missed > 0.10:
            print(f"STATUS:CRASH|REASON:Missed speech too high ({missed:.4f})")
            sys.exit(1)

        print(f"STATUS:SUCCESS|DER:{total_der:.4f}|MS:{missed:.4f}|FA:{false_alarm:.4f}|CF:{confusion:.4f}")
        
    except Exception as e:
        print(f"STATUS:CRASH|REASON:{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()