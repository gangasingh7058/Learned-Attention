import torch
from torch.utils.data import Dataset
import tiktoken

class Shakes_Pear_Dataset(Dataset):
    
    def __init__(self, seq_len):
        self.enc = tiktoken.get_encoding("gpt2")  # BPE tokenizer (~50K vocab)

        paths = [
            "../Datasets/Shakespear_dataset/Shakes_Pear.txt"
        ]

        raw_text = ""
        for path in paths:
            print(f"Reading File : {path}")
            with open(path, 'r', encoding='utf-8') as f:
                raw_text += f.read() + "\n"

        # Tokenize with BPE
        self.tokens = self.enc.encode(raw_text)

        self.vocab_size = self.enc.n_vocab  # 50257 for GPT-2 encoding
        self.seq_len = seq_len
        self.num_samples = len(self.tokens) - seq_len

    def get_vocab_size(self):
        return self.vocab_size

    def get_mask(self):
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len))
        return mask.unsqueeze(0) == 1  # (1, seq_len, seq_len) — DataLoader adds batch dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        src    = torch.tensor(chunk[:-1], dtype=torch.long)   # tokens[idx : idx+seq_len]
        target = torch.tensor(chunk[1:],  dtype=torch.long)   # tokens[idx+1 : idx+seq_len+1]

        return {
            "src": src,
            "target": target,
            "mask": self.get_mask(),
        }

