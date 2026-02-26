import torch
from torch.utils.data import Dataset
import requests
import zipfile
import io
from collections import Counter


class WikiPedia_Dataset(Dataset):

    def __init__(self, seq_len: int, mode: str = "train", test_fraction: float = 0.1):
        self.mode = mode
        self.seq_len = seq_len

        # Download text8 dataset
        url = 'http://mattmahoney.net/dc/text8.zip'
        print(f"Loading Data from => {url}")
        r = requests.get(url)
        f = zipfile.ZipFile(io.BytesIO(r.content))
        text = f.read('text8').decode('utf-8')

        tokens = text.split()

        # Build vocabulary with UNK
        counter = Counter(tokens)
        min_freqs = 10

        vocab = {"<UNK>": 0}
        for word, freq in counter.items():
            if freq >= min_freqs:
                vocab[word] = len(vocab)

        self.vocab = vocab
        self.vocab_size = len(vocab)

        # Convert tokens to ids
        all_tokens = [vocab.get(token, 0) for token in tokens]

        # Sequential split (important for LM)
        split_idx = int(len(all_tokens) * (1 - test_fraction))
        self.train_tokens = all_tokens[:2_000_000]
        self.test_tokens = all_tokens[split_idx:]

        # Prevent negative length bug
        self.num_samples_train = max(0, len(self.train_tokens) - seq_len)
        self.num_samples_test = max(0, len(self.test_tokens) - seq_len)

        # Create causal mask once
        self.mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0) == 1

    def get_vocab_size(self):
        return self.vocab_size

    def get_mask(self):
        return self.mask

    def __len__(self):
        if self.mode == "train":
            return self.num_samples_train
        else:
            return self.num_samples_test

    def __getitem__(self, idx):

        if self.mode == "train":
            chunk = self.train_tokens[idx: idx + self.seq_len + 1]
        else:
            chunk = self.test_tokens[idx: idx + self.seq_len + 1]

        src = torch.tensor(chunk[:-1], dtype=torch.long)
        target = torch.tensor(chunk[1:], dtype=torch.long)

        return {
            "src": src,
            "target": target,
            "mask": self.mask,
        }