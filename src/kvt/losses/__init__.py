from .bce import BCEWithLogitsLossAndIgnoreIndex
from .combo import SegmentationWithClassificationHeadLoss
from .dice import DiceLoss
from .flood import FloodingBCEWithLogitsLoss
from .focal import (
    BinaryDualFocalLoss,
    BinaryFocalLoss,
    BinaryReducedFocalLoss,
    FocalLoss,
    LabelSmoothBinaryFocalLoss,
)
from .lovasz import LovaszHingeLoss, LovaszSoftmaxLoss
from .noisy import (
    IterativeSelfLearningLoss,
    JointOptimizationLoss,
    LabelSmoothingCrossEntropy,
    OUSMLoss,
    SymmetricBCELoss,
    SymmetricBinaryFocalLoss,
    SymmetricCrossEntropy,
    coral_loss,
)
from .ohem import OHEMLoss, OHEMLossWithLogits
from .rmse import RMSELoss
from .ssl import DDINOLoss, DINOLoss
from .vat import VATLoss
