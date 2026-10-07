import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelLoader:
    def __init__(self, model_name: str, device: str = "cuda", torch_dtype="auto"):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.tokenizer = None
        self.model = None
        self.config = None

    def load_model(self):
        """
        Load the tokenizer and model from the specified model name.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
            )
            if self.device is not None:
                self.model.to(self.device)
            self.config = self.model.config
            dtype = next(self.model.parameters()).dtype
            print(
                f"Successfully loaded model: {self.model_name} "
                f"(device={self.device}, dtype={dtype})"
            )
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")

    def get_tokenizer(self):
        """
        Get the loaded tokenizer.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        return self.tokenizer

    def get_model(self):
        """
        Get the loaded model.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model

    def get_config(self):
        """
        Get the configuration of the loaded model.
        """
        if self.config is None:
            raise ValueError("Config not loaded. Call load_model() first.")
        return self.config