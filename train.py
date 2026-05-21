import os
import torch
import pytorch_lightning as pl
from pyannote.audio.tasks import Segmentation
from pyannote.audio.models.segmentation import PyanNet
from pyannote.database import ProtocolFile, Registry
from pyannote.database.protocol.speaker_diarization import SpeakerDiarizationProtocol
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
from agent_config import get_agent_configuration

# 1. Define a Mock Protocol for the single POD_711 file
class SingleFileProtocol(SpeakerDiarizationProtocol):
    def trn_iter(self):
        rttm_data = load_rttm("POD_711.rttm")
        annotation = list(rttm_data.values())[0]
        yield ProtocolFile({"uri": "POD_711", "audio": "POD_711.wav", "annotation": annotation})
    
    def dev_iter(self):
        yield from self.trn_iter()
        
    def tst_iter(self):
        yield from self.trn_iter()

def main():
    # Load agent configurations
    hparams, augmentation = get_agent_configuration()
    
    # Setup Pyannote Task
    protocol = SingleFileProtocol(name="POD_711_Protocol")
    task = Segmentation(
        protocol, 
        duration=5.0, # 5-second audio chunks
        max_num_speakers=3, 
        batch_size=hparams.get("batch_size", 8),
        num_workers=4,
        loss="bce",
        augmentation=augmentation
    )

    # Load Base Model
    model = PyanNet(
        sincnet={"stride": 10}, 
        task=task
    )
    model.setup(stage="fit")

    # Force learning rate injection
    model.learning_rate = hparams.get("learning_rate", 1e-4)

    # Setup Trainer (Strictly 2 epochs for rapid loop testing)
    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False, # We save manually
        logger=False
    )

    # Execute Fine-Tuning
    print("Starting Fine-Tuning...")
    trainer.fit(model)

    # Save Checkpoint
    os.makedirs("./best_model", exist_ok=True)
    torch.save(model.state_dict(), "./best_model/pytorch_model.bin")
    print("Model saved to ./best_model/pytorch_model.bin")

if __name__ == "__main__":
    main()