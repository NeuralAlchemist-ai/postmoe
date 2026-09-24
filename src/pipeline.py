from postmoe.eval import Evaluation
from postmoe.extractor import UniversalAttentionDecoupler
from postmoe.converter import Converter
from .postmoe.MLA import MLA_Attention

from loader.load_model import ModelLoader
from loader.data_loader import DataLoader

class Pipeline:
    def __init__(self, config):

        self.UniversalAttentionDecoupler = UniversalAttentionDecoupler
        self.Converter = Converter
        self.Evaluation = Evaluation
        self.MLA_Attention = MLA_Attention

        self.ModelLoader = ModelLoader
        self.DataLoader = DataLoader

        self.model_name = config.model_name
        self.rope_dim = config.rope_dim
        self.dataset_name = config.dataset_name
        self.dataset_config = config.dataset_config
        self.split = config.split

        self.model = None
        self.tokenizer = None
        self.config = None
        self.data = None

    def convert(self):
        model_loader = self.ModelLoader(self.model_name)
        model_loader.load_model()
        self.model, self.tokenizer, self.config = model_loader.get_model(), model_loader.get_tokenizer(), model_loader.get_config()

        data_loader = self.DataLoader(
            self.dataset_name,
            self.dataset_config,
            self.split,
        )
        self.data = data_loader.load_data()

    def custom_MLA_attention(self, o_proj, idx):

        W_k_nope_target, W_k_rope_target, W_q_nope, W_q_rope, W_v = self.UniversalAttentionDecoupler(
            self.model,
            self.config,
            self.rope_dim,
        ).convert_to_target_layout(repeat_mode="gqa_broadcast")
        k_up_proj, kv_down_proj = self.Converter(W_k_nope_target).svd_compress()
        v_up_proj, _ = self.Converter(W_v).svd_compress()


        return self.MLA_Attention(
            kv_down_proj=kv_down_proj,
            k_up_proj=k_up_proj,
            v_up_proj=v_up_proj,
            q_nope=W_q_nope,
            q_rope=W_q_rope,
            k_rope=W_k_rope_target,
            o_proj=o_proj,
            idx = idx
        ) 


    def evaluate(self):
        try:
            vram, perplexity = self.Evaluation(self.model, self.tokenizer, self.data, self.compressed_nope).evaluate()
            print(f"VRAM Usage: {vram} GB")
            print(f"Perplexity: {perplexity}")
        except Exception as e:
            print(f"Error during evaluation: {e}")

