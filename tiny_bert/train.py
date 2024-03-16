# train a tiny 2-layer, attention-only "bert" model
# It's just Neel's tiny gpt model but with bidirectional attention.
# the loss fn is bert's masked language model loss.

import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from datasets import load_dataset
from model import get_config_and_tok


def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(tokenizer.vocab_size, labels.shape, dtype=torch.long, device=inputs.device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(model: HookedTransformer, dataloader: DataLoader, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for batch in dataloader:
        inputs = batch["tokens"].to(device)

        inputs, labels = mask_tokens(inputs, model.tokenizer)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Bert's masked language model loss
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, model.tokenizer.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        
        loss.backward()
        print(loss.item())
        optimizer.step()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, tokenizer = get_config_and_tok(bidirectional=True)
    model = HookedTransformer(config, tokenizer, move_to_device=False)
    model.to(device)

    # data is already tokenized and is in chunks of 1024 token ids.
    data = load_dataset("NeelNanda/c4-code-tokenized-2b", split="train[:10%]")
    data.set_format(type="torch", columns=["tokens"])
    print('loaded data')

    dataloader = DataLoader(data, batch_size=32, shuffle=True)
    print('initialized dataloader')

    train(model, dataloader, device)

if __name__ == "__main__":
    main()






