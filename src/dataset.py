import torch
from torch.utils.data import Dataset

class Shakes_Pear_Dataset(Dataset):
    
    def __init__(self, seq_len: int, tokenizer, mode: str = "train", test_fraction: float = 0.1):
        self.enc = tokenizer
        self.mode = mode    
        
        paths = [
                "../Datasets/Shakespear_dataset/Shakes_Pear.txt"
            ]        

        raw_text = ""
        for path in paths:
            print(f"Reading File : {path}")
            with open(path, 'r', encoding='utf-8') as f:
                raw_text += f.read() + "\n"

        # Tokenize with BPE
        all_tokens = self.enc.encode(raw_text)

        # Contiguous split: first 90% train, last 10% test
        # This preserves sequential order (critical for language modeling)
        split_idx = int(len(all_tokens) * (1 - test_fraction))
        self.train_tokens = all_tokens[:split_idx]
        self.test_tokens  = all_tokens[split_idx:]

        self.vocab_size = self.enc.vocab_size()  # SentencePiece vocab size
        self.seq_len = seq_len
        self.num_samples_train = len(self.train_tokens) - seq_len
        self.num_samples_test = len(self.test_tokens) - seq_len

    def get_vocab_size(self):
        return self.vocab_size

    def get_mask(self):
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len))
        return mask.unsqueeze(0) == 1  # (1, seq_len, seq_len) — DataLoader adds batch dim

    def __len__(self):
        if self.mode == "train":
            return self.num_samples_train
        else:
            return self.num_samples_test    

    def __getitem__(self, idx):

        if self.mode == "train":
            chunk = self.train_tokens[idx : idx + self.seq_len + 1]
        else:
            chunk = self.test_tokens[idx : idx + self.seq_len + 1]

        src    = torch.tensor(chunk[:-1], dtype=torch.long)   # tokens[idx : idx+seq_len]
        target = torch.tensor(chunk[1:],  dtype=torch.long)   # tokens[idx+1 : idx+seq_len+1]

        return {
            "src": src,
            "target": target,
            "mask": self.get_mask(),
        }


