from fmengine.modelling._constants import CONFIG_NAME

class ModelMixin():
    """Mixin class for pretrained models."""
    config_name = CONFIG_NAME
    _automatically_saved_args = ["_diffusers_version", "_class_name", "_name_or_path"]
    _flax_internal_args = ["name", "parent", "dtype"]