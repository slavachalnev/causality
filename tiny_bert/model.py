import torch
import transformer_lens
from transformer_lens import HookedTransformer


def get_config_and_tok(bidirectional=True):
    causal_model = HookedTransformer.from_pretrained("attn-only-2l")
    config = causal_model.cfg.to_dict()

    config['init_weights'] = True

    config['attention_dir'] = 'bidirectional' if bidirectional else 'causal'
    return config, causal_model.tokenizer






