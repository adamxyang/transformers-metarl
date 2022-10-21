"""PyTorch Policies."""
from garage.torch.policies.categorical_cnn_policy import CategoricalCNNPolicy
from garage.torch.policies.context_conditioned_policy import (
    ContextConditionedPolicy)
from garage.torch.policies.deterministic_mlp_policy import (
    DeterministicMLPPolicy)
from garage.torch.policies.discrete_qf_argmax_policy import (
    DiscreteQFArgmaxPolicy)
from garage.torch.policies.gaussian_mlp_policy import GaussianMLPPolicy
from garage.torch.policies.policy import Policy
from garage.torch.policies.tanh_gaussian_mlp_policy import (
    TanhGaussianMLPPolicy)

from garage.torch.policies.gaussian_transformer_policy import GaussianTransformerPolicy
from garage.torch.policies.gaussian_transformer_encoder_policy import GaussianTransformerEncoderPolicy
from garage.torch.policies.beta_transformer_encoder_policy import BetaTransformerEncoderPolicy
from garage.torch.policies.memory_transformer.gaussian_memory_transformer_policy import GaussianMemoryTransformerPolicy

__all__ = [
    'CategoricalCNNPolicy',
    'DeterministicMLPPolicy',
    'DiscreteQFArgmaxPolicy',
    'GaussianMLPPolicy',
    'Policy',
    'TanhGaussianMLPPolicy',
    'ContextConditionedPolicy',
    'GaussianTransformerPolicy',
    'GaussianTransformerEncoderPolicy',
    'GaussianMemoryTransformerPolicy',
    'BetaTransformerEncoderPolicy',
]
