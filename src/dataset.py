import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    """
    Translation dataset for a decoder-only Transformer with separate
    source and target vocabularies.

    Input (src):  source language tokens padded to seq_len
    Label (tgt):  target language tokens padded to seq_len

    The model receives src token IDs (embedded via src_vocab),
    and predicts tgt token IDs (projected via tgt_vocab).
    """

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # Source special token IDs
        self.src_pad_id = tokenizer_src.token_to_id("[PAD]")

        # Target special token IDs
        self.tgt_sos_id = tokenizer_tgt.token_to_id("[SOS]")
        self.tgt_eos_id = tokenizer_tgt.token_to_id("[EOS]")
        self.tgt_pad_id = tokenizer_tgt.token_to_id("[PAD]")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        pair = self.ds[index]
        src_text = pair['translation'][self.src_lang]
        tgt_text = pair['translation'][self.tgt_lang]

        # Tokenize
        src_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Truncate to fit within seq_len
        # src: tokens + padding → seq_len
        # tgt input:  [SOS] + tokens → seq_len  (teacher forcing input)
        # tgt label:  tokens + [EOS] → seq_len  (what model should predict)
        src_tokens = src_tokens[:self.seq_len]
        tgt_tokens = tgt_tokens[:self.seq_len - 1]  # leave room for SOS/EOS

        src_pad_len = self.seq_len - len(src_tokens)
        tgt_pad_len = self.seq_len - len(tgt_tokens) - 1  # -1 for SOS or EOS

        # Source: src_tokens [PAD]...
        src = torch.cat([
            torch.tensor(src_tokens, dtype=torch.int64),
            torch.tensor([self.src_pad_id] * src_pad_len, dtype=torch.int64),
        ])

        # Target label: tgt_tokens [EOS] [PAD]...
        # This is what the model should predict at each position
        tgt = torch.cat([
            torch.tensor(tgt_tokens, dtype=torch.int64),
            torch.tensor([self.tgt_eos_id], dtype=torch.int64),
            torch.tensor([self.tgt_pad_id] * tgt_pad_len, dtype=torch.int64),
        ])

        assert src.size(0) == self.seq_len
        assert tgt.size(0) == self.seq_len

        return {
            "src": src,        # (Seq_Len) — source language token IDs
            "tgt": tgt,        # (Seq_Len) — target language token IDs (label)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
