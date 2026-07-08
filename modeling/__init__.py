from .rpn import SBRPN
from .backbone import ClipRN101
from .meta_arch_COT3_30 import ClipRCNNWithClipBackbone
from .meta_arch_newCOT import ClipRCNNSECOT  # noqa: F401
from .roi_head import ClipRes5ROIHeads
from .config import add_stn_config
from .custom_pascal_evaluation import CustomPascalVOCDetectionEvaluator