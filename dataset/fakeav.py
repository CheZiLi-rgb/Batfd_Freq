import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Callable, Any, Union, Tuple

import einops
import numpy as np
import torch
import torchaudio
from einops import rearrange
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import Tensor
from torch.nn import functional as F, Identity
from torch.utils.data import DataLoader, RandomSampler, Dataset

from utils import read_json, read_video, padding_video, padding_audio, resize_video, iou_with_anchors, ioa_with_anchors


@dataclass
class Metadata:
    file: str
    n_fakes: int
    fake_periods: List[List[int]]
    duration: float
    original: Optional[str]
    modify_video: bool
    modify_audio: bool
    split: str
    video_frames: int
    audio_channels: int
    audio_frames: int


T_LABEL = Union[Tensor, Tuple[Tensor, Tensor, Tensor]]


class FakeAVDataset(Dataset):
    def __init__(self, subset: str, root: str = "/public/home/lc_211124030061", frame_padding: int = 512,
                 max_duration: int = 40, fps: int = 25,
                 video_transform: Callable[[Tensor], Tensor] = Identity(),
                 audio_transform: Callable[[Tensor], Tensor] = Identity(),
                 metadata: Optional[List[Metadata]] = None,
                 get_meta_attr: Callable[[Metadata, Tensor, Tensor, T_LABEL], List[Any]] = None,
                 require_match_scores: bool = False,
                 return_file_name: bool = False,
                 metadata_name: str = "metadata_min.json"
                 ):
        self.subset = subset
        self.root = root
        self.video_padding = frame_padding
        self.audio_padding = int(frame_padding / fps * 16000)
        self.max_duration = max_duration
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.get_meta_attr = get_meta_attr
        self.require_match_scores = require_match_scores
        self.return_file_name = return_file_name

        if metadata is None:
            metadata: List[Metadata] = read_json(os.path.join(self.root, metadata_name), lambda x: Metadata(**x))
            self.metadata: List[Metadata] = [each for each in metadata if each.split == subset]

        else:
            self.metadata: List[Metadata] = metadata

        print(f"Load {metadata_name} successfully.")
        print(f"Load {len(self.metadata)} data in {subset}.")

    def __getitem__(self, index: int) -> List[Tensor]:
        try:
            meta = self.metadata[index]

            video, audio, _ = read_video(meta.file)
            
            video = padding_video(video, target=self.video_padding)
            audio = padding_audio(audio, target=self.audio_padding)

            video = self.video_transform(video)
            audio = self.audio_transform(audio)

            video = rearrange(resize_video(video, (96, 96)), "t c h w -> c t h w")
            audio = self._get_log_mel_spectrogram(audio)

            label = self.get_label(meta)
            outputs = [video, audio, label]

            if self.return_file_name:
                outputs.append(meta.file)

            return outputs

        except IndexError:
            print(f"Index {index} out of range for metadata list.")
            return torch.zeros(1)  # Return an empty tensor instead of None

        except FileNotFoundError as e:
            print(f"File not found: {os.path.join(meta.file)}")
            return torch.zeros(1)  # Return an empty tensor instead of None

        except Exception as e:
            print("An error occurred while processing the item.")
            return torch.zeros(1)  # Return an empty tensor instead of None

    def get_label(self, meta: Metadata) -> Tensor:
        # meta.file:"/public/liche/FakeAVCeleb_v1.2/RealVideo-RealAudio/African/men/id00166/00010.mp4"
        if meta.n_fakes == 0:
            label = 0
        else:
            label = 1
        label_tensor = torch.tensor([float(label)], dtype=torch.float32)
        return label_tensor

    def __len__(self) -> int:
        return len(self.metadata)

    @staticmethod
    def _get_log_mel_spectrogram(audio: Tensor) -> Tensor:
        ms = torchaudio.transforms.MelSpectrogram(n_fft=321, n_mels=64)
        spec = torch.log(ms(audio[:, 0]) + 0.01)
        return spec


def _default_get_meta_attr(meta: Metadata, video: Tensor, audio: Tensor, label: Tensor) -> List[Any]:
    return [meta.video_frames]


class FakeAVDataModule(LightningDataModule):
    train_dataset: FakeAVDataset
    dev_dataset: FakeAVDataset
    test_dataset: FakeAVDataset
    metadata: List[Metadata]

    def __init__(self, root: str = "/public/home/lc_211124030061/FakeAVCeleb_v1.2/", frame_padding=512, max_duration=45,
                 feature_types: Tuple[Optional[str], Optional[str]] = (None, None),
                 batch_size: int = 1, num_workers: int = 0,
                 take_train: int = None, take_dev: int = None, take_test: int = None,
                 cond: Optional[Callable[[Metadata], bool]] = None,
                 get_meta_attr: Callable[[Metadata, Tensor, Tensor, Tensor], List[Any]] = _default_get_meta_attr,
                 return_file_name: bool = False,
                 metadata_name: str = "metadata_min.json"
                 ):
        super().__init__()
        self.root = root
        self.frame_padding = frame_padding
        self.max_duration = max_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.take_train = take_train
        self.take_dev = take_dev
        self.take_test = take_test
        self.cond = cond
        self.get_meta_attr = get_meta_attr
        self.return_file_name = return_file_name
        self.Dataset = feature_type_to_dataset_type[feature_types]
        self.metadata_name = metadata_name

    def setup(self, stage: Optional[str] = None) -> None:
        self.metadata: List[Metadata] = read_json(os.path.join(self.root, self.metadata_name), lambda x: Metadata(**x))

        train_metadata = []
        dev_metadata = []
        test_metadata = []

        for meta in self.metadata:
            if self.cond is None or self.cond(meta):
                if meta.split == "train":
                    train_metadata.append(meta)
                elif meta.split == "none":
                    dev_metadata.append(meta)
                elif meta.split == "test":
                    test_metadata.append(meta)

        if self.take_dev is not None:
            dev_metadata = dev_metadata[:self.take_dev]

        self.train_dataset = self.Dataset("train", self.root, self.frame_padding, self.max_duration,
                                          metadata=train_metadata, get_meta_attr=self.get_meta_attr,
                                          return_file_name=self.return_file_name, metadata_name=self.metadata_name
                                          )
        self.dev_dataset = self.Dataset("dev", self.root, self.frame_padding, self.max_duration,
                                        metadata=dev_metadata, get_meta_attr=self.get_meta_attr,
                                        return_file_name=self.return_file_name, metadata_name=self.metadata_name
                                        )
        self.test_dataset = self.Dataset("test", self.root, self.frame_padding, self.max_duration,
                                         metadata=test_metadata, get_meta_attr=self.get_meta_attr,
                                         return_file_name=self.return_file_name, metadata_name=self.metadata_name
                                         )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          sampler=RandomSampler(self.train_dataset, num_samples=self.take_train, replacement=True)
                          )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


# The dictionary is used to map the feature type to the dataset type
# The key is a tuple of (visual_feature_type, audio_feature_type), ``None`` means using end-to-end encoder.
feature_type_to_dataset_type = {
    (None, None): FakeAVDataset
}
