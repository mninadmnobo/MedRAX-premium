# Permanent fix: add_model_info_to_auto_map was removed in newer transformers.
# LlavaMistralForCausalLM.__init_subclass__ calls it, crashing on import.
# This monkey-patch prevents that crash.
import transformers.utils
if not hasattr(transformers.utils, 'add_model_info_to_auto_map'):
    transformers.utils.add_model_info_to_auto_map = lambda *args, **kwargs: None

from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
