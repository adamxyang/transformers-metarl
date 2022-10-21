"""PyTorch Modules."""
# yapf: disable
# isort:skip_file
from garage.torch.modules.categorical_cnn_module import CategoricalCNNModule
from garage.torch.modules.cnn_module import CNNModule
from garage.torch.modules.gaussian_mlp_module import (
    GaussianMLPIndependentStdModule)  # noqa: E501
from garage.torch.modules.gaussian_mlp_module import (
    GaussianMLPTwoHeadedModule)  # noqa: E501
from garage.torch.modules.gaussian_mlp_module import GaussianMLPModule

from garage.torch.modules.beta_mlp_module import (
    BetaMLPIndependentStdModule)  # noqa: E501
from garage.torch.modules.beta_mlp_module import (
    BetaMLPTwoHeadedModule)  # noqa: E501
from garage.torch.modules.beta_mlp_module import BetaMLPModule

from garage.torch.modules.mlp_module import MLPModule
from garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule
# DiscreteCNNModule must go after MLPModule
from garage.torch.modules.discrete_cnn_module import DiscreteCNNModule
from garage.torch.modules.transformer_no_ln import TransformerEncoderLayerNoLN
# yapf: enable

__all__ = [
    'CategoricalCNNModule',
    'CNNModule',
    'DiscreteCNNModule',
    'MLPModule',
    'MultiHeadedMLPModule',
    'GaussianMLPModule',
    'GaussianMLPIndependentStdModule',
    'GaussianMLPTwoHeadedModule',
    'TransformerEncoderLayerNoLN',
    'BetaMLPModule',
    'BetaMLPIndependentStdModule',
    'BetaMLPTwoHeadedModule',
]
