import sys
import warnings
import os
import torch
import yaml
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.audio import Pipeline
from pyannote.audio.models.segmentation import PyanNet
from agent_config import get_agent_configuration

warnings.filterwarnings("ignore")

def main():
    truth_rttm = "POD_711.rttm"
    audio_file = "POD_711.wav"
    model_checkpoint = "./best_model/pytorch_model.bin"
    
    hparams, _ = get_agent_configuration()
    
    try:
        truth_annotations = list(load_rttm(truth_rttm).values())[0]
        
        # 1. Load Base Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=os.environ.get("HF_TOKEN")
        )
        
        # 2. Inject Custom Fine-Tuned Weights into the Pipeline's Segmentation Model
        if os.path.exists(model_checkpoint):
            custom_state_dict = torch.load(model_checkpoint, map_location="cpu")
            pipeline._segmentation.model.load_state_dict(custom_state_dict)
        else:
            raise FileNotFoundError("Fine-tuned model checkpoint not found.")

        pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 3. Apply Agent VAD Hyperparameters
        pipeline.instantiate({
            "segmentation": {
                "onset": hparams.get("vad_onset", 0.5),
                "offset": hparams.get("vad_offset", 0.5),
                "min_duration_on": hparams.get("min_speech_duration", 0.1),
            }
        })
        
        # 4. Run Inference
        pred_annotation = pipeline(audio_file)
        
        metric = DiarizationErrorRate()
        der = metric(truth_annotations, pred_annotation, detailed=True)
        
        total_speech = der['total']
        missed = der['missed detection'] / total_speech
        false_alarm = der['false alarm'] / total_speech
        confusion = der['confusion'] / total_speech
        total_der = der['diarization error rate']

        if missed > 0.10:
            print(f"STATUS:CRASH|REASON:Missed speech too high ({missed:.4f})")
            sys.exit(1)

        print(f"STATUS:SUCCESS|DER:{total_der:.4f}|MS:{missed:.4f}|FA:{false_alarm:.4f}|CF:{confusion:.4f}")
        
    except Exception as e:
        print(f"STATUS:CRASH|REASON:{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()