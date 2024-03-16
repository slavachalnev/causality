import torch
import transformer_lens
from transformer_lens import HookedTransformer


def get_config_and_tok(bidirectional=True):
    causal_model = HookedTransformer.from_pretrained("attn-only-2l")
    config = causal_model.cfg.to_dict()
    config['init_weights'] = True
    config['attention_dir'] = 'bidirectional' if bidirectional else 'causal'

    config['d_vocab'] = config['d_vocab'] + 1 # add mask token
    tokenizer = causal_model.tokenizer
    tokenizer.add_special_tokens({'mask_token': "<MASK>"}) # tok id 48262

    return config, tokenizer


# config, tok = get_config_and_tok()
# print('done')
