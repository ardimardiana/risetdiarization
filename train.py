import os
import warnings
import torch
import pytorch_lightning as pl
from pyannote.audio import Model
from pyannote.audio.tasks import SpeakerDiarization
from pyannote.database.protocol.speaker_diarization import SpeakerDiarizationProtocol
from pyannote.database.util import load_rttm
from agent_config import get_agent_configuration

# Mute the torch_audiomentations FutureWarnings to keep stdout clean
warnings.filterwarnings("ignore")

# 1. Define a Mock Protocol for the single POD_711 file
class SingleFileProtocol(SpeakerDiarizationProtocol):
    def trn_iter(self):
        rttm_data = load_rttm("POD_711.rttm")
        annotation = list(rttm_data.values())[0]
        # Yield a standard dictionary directly
        yield {"uri": "POD_711", "audio": "POD_711.wav", "annotation": annotation}
    
    def dev_iter(self):
        yield from self.trn_iter()
        
    def tst_iter(self):
        yield from self.trn_iter()

def main():
    hparams, augmentation = get_agent_configuration()
    
    # Initialize without the unsupported 'name' argument
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
        "pyannote/segmentation-3.1", 
        token=hf_token
    )
    
    model.task = task
    model.setup(stage="fit")

    # Force learning rate injection
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