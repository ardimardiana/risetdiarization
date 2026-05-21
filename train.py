import os
import warnings
import torch
import torchaudio
import pytorch_lightning as pl
from pyannote.core import Segment, Timeline
from pyannote.audio import Model
from pyannote.audio.tasks import SpeakerDiarization
from pyannote.database.protocol.speaker_diarization import SpeakerDiarizationProtocol
from pyannote.database.util import load_rttm
from agent_config import get_agent_configuration

warnings.filterwarnings("ignore")

class SingleFileProtocol(SpeakerDiarizationProtocol):
    # 1. FIX: Tell Pyannote that speaker labels are unique to the file
    scope = "file" 

    def trn_iter(self):
        rttm_data = load_rttm("POD_711.rttm")
        annotation = list(rttm_data.values())[0]
        
        # 2. FIX: Create an 'annotated' timeline so the Task knows where to sample chunks
        info = torchaudio.info("POD_711.wav")
        duration = info.num_frames / info.sample_rate
        annotated = Timeline([Segment(0, duration)])

        yield {
            "uri": "POD_711", 
            "audio": "POD_711.wav", 
            "annotation": annotation,
            "annotated": annotated
        }
    
    def dev_iter(self):
        yield from self.trn_iter()
        
    def tst_iter(self):
        yield from self.trn_iter()

def main():
    hparams, augmentation = get_agent_configuration()
    
    protocol = SingleFileProtocol()
    
    task = SpeakerDiarization(
        protocol, 
        duration=10.0,
        max_speakers_per_chunk=3,
        max_speakers_per_frame=2,
        batch_size=hparams.get("batch_size", 8),
        num_workers=4,
        augmentation=augmentation
    )

    hf_token = os.environ.get("HF_TOKEN")
    model = Model.from_pretrained(
        "pyannote/segmentation-3.0", 
        use_auth_token=hf_token
    )
    
    model.task = task
    model.setup(stage="fit")

    from types import MethodType
    from torch.optim import Adam
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=hparams.get("learning_rate", 1e-4))
        
    model.configure_optimizers = MethodType(configure_optimizers, model)

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False, 
        logger=False
    )

    print("Starting Fine-Tuning...")
    trainer.fit(model)

    os.makedirs("./best_model", exist_ok=True)
    torch.save(model.state_dict(), "./best_model/pytorch_model.bin")
    print("Model saved to ./best_model/pytorch_model.bin")

if __name__ == "__main__":
    main()