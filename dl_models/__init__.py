"""
Deep learning package (split layout):

- ``dl_models.algorithm`` — ``se_resnet_1d`` (1D CNN + SE-ResNet), ``multimodal_net`` (fusion + head); ``networks`` re-exports.
- ``dl_models.data`` — PyTorch ``Dataset``, CSV loading, feature names.
- ``dl_models.training`` — ``Trainer``, augmentation.
- ``dl_models.inference`` — ``DLDetector``.

Imports below match the previous flat layout so older paths keep working.
"""

from dl_models.algorithm import (
    AttentionFusion,
    DenseClassifier,
    ModalitySEResNet1D,
    MultimodalPDNet,
    ResidualSEBlock1D,
    SEBlock1D,
)
from dl_models.data import (
    GAIT_FEATURE_NAMES,
    HANDWRITING_FEATURE_NAMES,
    SPEECH_FEATURE_NAMES,
    MultimodalPDDataset,
    load_all_modalities,
    load_modality_csv,
    load_multimodal_feature_spec,
)
from dl_models.inference import DLDetector
from dl_models.training import Trainer, augment_batch

__all__ = [
    "SEBlock1D",
    "ResidualSEBlock1D",
    "ModalitySEResNet1D",
    "AttentionFusion",
    "DenseClassifier",
    "MultimodalPDNet",
    "SPEECH_FEATURE_NAMES",
    "HANDWRITING_FEATURE_NAMES",
    "GAIT_FEATURE_NAMES",
    "MultimodalPDDataset",
    "load_multimodal_feature_spec",
    "load_modality_csv",
    "load_all_modalities",
    "Trainer",
    "augment_batch",
    "DLDetector",
]
