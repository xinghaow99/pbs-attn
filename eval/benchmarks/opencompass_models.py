# Custom OpenCompass model classes for patched models
import os
import sys
from typing import Optional, Callable


current_dir = os.path.dirname(os.path.abspath(__file__))
abs_path = os.path.join(current_dir, '../../pbs_attn')
if abs_path not in sys.path:
    sys.path.insert(0, abs_path)

from opencompass.models.huggingface_above_v4_33 import HuggingFacewithChatTemplate
from opencompass.registry import MODELS
from pbs_attn.patch.huggingface import (
    apply_patch_with_prefill,
    get_meanpooling_prefill,
    get_minference_prefill,

    get_xattention_prefill,
    get_flexprefill_prefill,
    get_flashattn_prefill,

    get_permuted_block_sparse_attn_fwd
)


@MODELS.register_module()
class PatchedHuggingFaceCausalLM(HuggingFacewithChatTemplate): 
    def __init__(self, 
                 path: str,
                 patch_type: str = 'meanpooling',
                 patch_kwargs: dict = {},
                 **kwargs):
        self.patch_type = patch_type
        self.patch_kwargs = patch_kwargs
        super().__init__(path=path, **kwargs)
    
    def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
        """Override _load_model to apply patch after model loading."""
        # First load the model normally
        super()._load_model(path, kwargs, peft_path, peft_kwargs)
        
        # Then apply the appropriate patch
        prefill_fn = self._get_prefill_function()
        if prefill_fn is not None:
            print(f"🔧 Applying {self.patch_type} patch to model...")
            self.model = apply_patch_with_prefill(self.model, prefill_fn)
        else:
            print("⚠️  No patch applied - using original model")
    
    def _get_prefill_function(self) -> Optional[Callable]:
        """Get the appropriate prefill function based on patch_type."""
        if self.patch_type == 'meanpooling':
            return get_meanpooling_prefill(**self.patch_kwargs)
        elif self.patch_type == 'minference':
            return get_minference_prefill(**self.patch_kwargs)
        elif self.patch_type == 'xattention':
            return get_xattention_prefill(**self.patch_kwargs)
        elif self.patch_type == 'flexprefill':
            return get_flexprefill_prefill(**self.patch_kwargs)
        elif self.patch_type == 'flashattn':
            return get_flashattn_prefill(**self.patch_kwargs)
        elif self.patch_type == 'pbs':
            return get_permuted_block_sparse_attn_fwd(**self.patch_kwargs)
        else:
            print(f"⚠️  Unknown patch_type: {self.patch_type}")
            
            return None


# Convenience classes for specific patches
@MODELS.register_module()
class MeanPoolingHuggingFaceCausalLM(PatchedHuggingFaceCausalLM):
    """HuggingFace CausalLM with MeanPooling sparse attention."""
    def __init__(self, path: str, **kwargs):
        patch_kwargs = kwargs.pop('patch_kwargs', {})
        super().__init__(path=path, patch_type='meanpooling', patch_kwargs=patch_kwargs, **kwargs)


@MODELS.register_module()
class MinferenceHuggingFaceCausalLM(PatchedHuggingFaceCausalLM):
    """HuggingFace CausalLM with Minference sparse attention."""
    def __init__(self, path: str, **kwargs):
        patch_kwargs = kwargs.pop('patch_kwargs', {})
        super().__init__(path=path, patch_type='minference', patch_kwargs=patch_kwargs, **kwargs)


@MODELS.register_module()
class ABSHuggingFaceCausalLM(PatchedHuggingFaceCausalLM):
    """HuggingFace CausalLM with ABS (Accurate Block Selection) sparse attention."""
    def __init__(self, path: str, **kwargs):
        patch_kwargs = kwargs.pop('patch_kwargs', {})
        super().__init__(path=path, patch_type='abs', patch_kwargs=patch_kwargs, **kwargs)


@MODELS.register_module()
class XAttentionHuggingFaceCausalLM(PatchedHuggingFaceCausalLM):
    """HuggingFace CausalLM with XAttention sparse attention."""
    def __init__(self, path: str, **kwargs):
        patch_kwargs = kwargs.pop('patch_kwargs', {})
        super().__init__(path=path, patch_type='xattention', patch_kwargs=patch_kwargs, **kwargs)


@MODELS.register_module()
class FlexPrefillHuggingFaceCausalLM(PatchedHuggingFaceCausalLM):
    """HuggingFace CausalLM with FlexPrefill sparse attention."""
    def __init__(self, path: str, **kwargs):
        patch_kwargs = kwargs.pop('patch_kwargs', {})
        super().__init__(path=path, patch_type='flexprefill', patch_kwargs=patch_kwargs, **kwargs) 


@MODELS.register_module()
class FlashAttnHuggingFaceCausalLM(PatchedHuggingFaceCausalLM):
    """HuggingFace CausalLM with FlashAttention."""
    def __init__(self, path: str, **kwargs):
        patch_kwargs = kwargs.pop('patch_kwargs', {})
        super().__init__(path=path, patch_type='flashattn', patch_kwargs=patch_kwargs, **kwargs)


@MODELS.register_module()
class PBSHuggingFaceCausalLM(PatchedHuggingFaceCausalLM):
    """HuggingFace CausalLM with Permuted Block Sparse (PBS) attention."""
    def __init__(self, path: str, **kwargs):
        patch_kwargs = kwargs.pop('patch_kwargs', {})
        super().__init__(path=path, patch_type='pbs', patch_kwargs=patch_kwargs, **kwargs)
