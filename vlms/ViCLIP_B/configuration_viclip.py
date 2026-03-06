from transformers import PretrainedConfig

class Config(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)