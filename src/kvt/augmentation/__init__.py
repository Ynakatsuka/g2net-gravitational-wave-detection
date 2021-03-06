from .audio import (
    CosineVolume,
    LowFrequencyMask,
    OneOf,
    PinkNoise,
    RandomVolume,
    SpecifiedNoise,
    SpeedTuning,
    StretchAudio,
    TimeShift,
)
from .augmix import RandomAugMix
from .autoaugment import ImageNetPolicy
from .block_fade import BlockFade
from .grid_mask import GridMask
from .histogram import HistogramNormalize
from .line import Line
from .morphological import RandomMorph
from .needless import NeedleAugmentation
from .random_erasing import RandomErasing
from .spec_augmentation import SpecAugmentationPlusPlus
from .sprinkle import Sprinkle
