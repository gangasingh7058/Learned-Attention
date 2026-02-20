import torch
from torch.utils.data import Dataset
from nltk import train_test_split

class Shakes_Pear_Dataset(Dataset):
    
    def __init__(self, seq_len: int,tokenizer,mode: str="train"):
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
        self.tokens = self.enc.encode(raw_text)

        self.train_tokens,self.test_tokens = train_test_split(self.tokens,test_size=0.1,random_state=42)

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

