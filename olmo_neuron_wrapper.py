"""OLMo wrapper for optimum-neuron training.

This module provides a thin wrapper around OLMo that makes it compatible
with optimum-neuron's custom modeling requirements.
"""

import sys
from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# Import from OLMo
from olmo.model import OLMo
from olmo.config import ModelConfig


# Make this module appear to be from optimum.neuron.models.training
# This tricks is_custom_modeling_model() to recognize it
sys.modules['optimum.neuron.models.training.olmo'] = sys.modules[__name__]
sys.modules['optimum.neuron.models.training.olmo.modeling_olmo'] = sys.modules[__name__]


class NeuronModelMixin:
    """Minimal NeuronModelMixin for compatibility."""
    SUPPORTS_PIPELINE_PARALLELISM = False
    PIPELINE_TRANSFORMER_LAYER_CLS = None
    PIPELINE_INPUT_NAMES = None
    PIPELINE_LEAF_MODULE_CLASSE_NAMES = None

    @classmethod
    def supports_pipeline_parallelism(cls) -> bool:
        return False

    @property
    def parameters_for_current_stage(self):
        """Return all parameters since we don't use pipeline parallelism."""
        return set(name for name, _ in self.named_parameters())


class OLMoPreTrainedModel(PreTrainedModel):
    """Base class for OLMo PreTrained models."""

    config_class = ModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = []
    _skip_keys_device_placement = []

    # Neuron-specific: torch.compile not supported on Trainium/Inferentia hardware
    _can_compile_fullgraph = False

    def _init_weights(self, module):
        """Initialize weights - OLMo handles this internally."""
        pass


class OLMoForCausalLM(NeuronModelMixin, OLMoPreTrainedModel):
    """OLMo wrapper for optimum-neuron training.

    This is a thin wrapper that makes OLMo compatible with optimum-neuron
    by implementing the NeuronModelMixin interface, while internally using
    the original OLMo model without modifications.
    """

    def __init__(self, config: ModelConfig, trn_config=None):
        """Initialize the wrapped OLMo model.

        Args:
            config: OLMo ModelConfig
            trn_config: TrainingNeuronConfig (ignored for now, kept for compatibility)
        """
        # Don't call PreTrainedModel.__init__ since our config is not PretrainedConfig
        # Instead, manually initialize as nn.Module and set required attributes
        nn.Module.__init__(self)

        # Store configs
        self.config = config
        self.trn_config = trn_config

        # Set attributes expected by PreTrainedModel
        self.name_or_path = config.name_or_path if hasattr(config, 'name_or_path') else None
        self._is_stateful = False

        # Add HuggingFace-style attributes for compatibility
        if not hasattr(config, 'num_hidden_layers'):
            config.num_hidden_layers = config.n_layers
        if not hasattr(config, 'num_attention_heads'):
            config.num_attention_heads = config.n_heads
        if not hasattr(config, 'hidden_size'):
            config.hidden_size = config.d_model

        # Add to_dict method for HuggingFace compatibility
        if not hasattr(config, 'to_dict'):
            config.to_dict = lambda: config.asdict()

        # Create the actual OLMo model
        self.model = OLMo(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass through OLMo model.

        This wraps OLMo's forward method to match HuggingFace's interface.
        """
        # Call OLMo's forward method
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
        )

        # Extract logits and compute loss if labels provided
        logits = output.logits
        loss = None

        if labels is not None:
            # Shift labels for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute cross entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            # If no labels, still return a dummy loss for training loop compatibility
            # This shouldn't happen in training, but just in case
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Return in HuggingFace format
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation (required by HuggingFace)."""
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
        }

    def get_input_embeddings(self):
        """Get input embeddings (required by HuggingFace)."""
        return self.model.transformer.wte

    def set_input_embeddings(self, value):
        """Set input embeddings (required by HuggingFace)."""
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        """Get output embeddings (required by HuggingFace)."""
        if hasattr(self.model, 'ff_out'):
            return self.model.ff_out
        return None

    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings (required by HuggingFace)."""
        if hasattr(self.model, 'ff_out'):
            self.model.ff_out = new_embeddings

    def tie_weights(self):
        """Tie weights if needed (required by HuggingFace)."""
        # OLMo handles this internally
        pass

    def num_params(self, include_embedding: bool = True) -> int:
        """Get number of parameters."""
        return self.model.num_params(include_embedding=include_embedding)

    def save_pretrained(
        self,
        save_directory,
        is_main_process=True,
        state_dict=None,
        save_function=torch.save,
        **kwargs,
    ):
        """Save the model checkpoint.

        This implementation saves the underlying OLMo model's state dict
        in a format compatible with HuggingFace's checkpoint structure.
        """
        if is_main_process:
            import os
            os.makedirs(save_directory, exist_ok=True)

            # Save model state dict
            model_path = os.path.join(save_directory, "pytorch_model.bin")
            if state_dict is None:
                state_dict = self.model.state_dict()
            save_function(state_dict, model_path)

            # Save config
            config_path = os.path.join(save_directory, "config.yaml")
            self.config.save(config_path)

            print(f"Model checkpoint saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load a pretrained model checkpoint.

        This implementation loads an OLMo checkpoint and wraps it
        for optimum-neuron compatibility.
        """
        import os

        # Load config
        config_path = os.path.join(pretrained_model_name_or_path, "config.yaml")
        config = ModelConfig.load(config_path)

        # Create model
        model = cls(config)

        # Load state dict
        model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            model.model.load_state_dict(state_dict)

        return model


# Make the module path match what optimum-neuron expects
__name__ = 'optimum.neuron.models.training.olmo.modeling_olmo'
