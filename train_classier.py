import argparse
import os
import torch

import toml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset.fakeav import FakeAVDataModule
from model.batfd_freq_classier import Batfd_Classifier
from utils import LrLogger, EarlyStoppingLR, generate_metadata_min

parser = argparse.ArgumentParser(description="BATFD training")
parser.add_argument("--config", type=str, default="./config/batfd_default_classier.toml")
parser.add_argument("--data_root", type=str, default="/data1/lic/FakeAVCeleb_v1.2")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--precision", default=16)
parser.add_argument("--num_train", type=int, default=None)
parser.add_argument("--num_val", type=int, default=50)
parser.add_argument("--max_epochs", type=int, default=25)
parser.add_argument("--resume", type=str, default=None)


if __name__ == '__main__':
    args = parser.parse_args()
    config = toml.load(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    learning_rate = config["optimizer"]["learning_rate"]
    gpus = args.gpus
    total_batch_size = args.batch_size * gpus
    learning_rate = learning_rate * total_batch_size / 4
    dataset = config["dataset"]

    v_encoder_type = config["model"]["video_encoder"]["type"]
    a_encoder_type = config["model"]["audio_encoder"]["type"]

    v_feature = None
    a_feature = None
    if config["model_type"] == "batfd":
        model = Batfd_Classifier(
            v_cla_feature_in=config["model"]["video_encoder"]["cla_feature_in"],
            a_cla_feature_in=config["model"]["audio_encoder"]["cla_feature_in"],
            weight_contrastive_loss=config["optimizer"]["contrastive_loss_weight"],
            contrast_loss_margin=config["optimizer"]["contrastive_loss_margin"],
            weight_decay=config["optimizer"]["weight_decay"],
            learning_rate=learning_rate,
            distributed=args.gpus > 1
        )
        require_match_scores = False
    else:
        raise ValueError("Invalid model type")

    if dataset == "FakeAVCeleb":
        dm = FakeAVDataModule(
            root=args.data_root,
            frame_padding=config["num_frames"],
            feature_types=(v_feature, a_feature),
            max_duration=config["max_duration"],
            batch_size=args.batch_size, num_workers=args.num_workers,
            take_train=args.num_train, take_dev=args.num_val,
        )
    else:
        raise ValueError("Invalid dataset type")

    try:
        precision = int(args.precision)
    except ValueError:
        precision = args.precision

    monitor = "val_loss"

    trainer = Trainer(log_every_n_steps=1, precision=precision, max_epochs=args.max_epochs,
                      callbacks=[
                          ModelCheckpoint(
                              dirpath=f"./ckpt/{config['name']}", save_last=True,
                              filename=config["name"] + "-{epoch}-{val_loss:.3f}",
                              monitor=monitor, mode="min"
                          ),
                          LrLogger(),
                          EarlyStoppingLR(lr_threshold=1e-7)
                      ], enable_checkpointing=True,
                      benchmark=True,
                      accelerator="auto",
                      devices=args.gpus,
                      strategy=None if args.gpus < 2 else "ddp",
                      resume_from_checkpoint=args.resume,
                      )

    trainer.fit(model, dm)
