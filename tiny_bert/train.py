# train a tiny 2-layer, attention-only "bert" model
# It's just Neel's tiny gpt model but with bidirectional attention.
# the loss fn is bert's masked language model loss.

import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from model import get_config_and_tok



def train(model: HookedTransformer, dataloader: DataLoader, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for batch in dataloader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Bert's masked language model loss
        loss = None # todo

        loss.backward()
        optimizer.step()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, tokenizer = get_config_and_tok(bidirectional=True)
    model = HookedTransformer(config, tokenizer, move_to_device=False)
    model.to(device)

    # Prepare your dataset and dataloader
    dataset = None  # Load your dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    train(model, dataloader, device)

if __name__ == "__main__":
    main()






