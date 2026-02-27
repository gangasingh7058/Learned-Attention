import torch
from torch.nn import functional as F
from model import Transformer
from model_args import ModelArgs
from pathlib import Path
from tokenizers import Tokenizer


def load_tokenizers(tokenizer_dir="../tokenizers/"):
    """Load the pre-trained source and target tokenizers."""
    tokenizer_src = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer_en.json"))
    tokenizer_tgt = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer_hi.json"))
    return tokenizer_src, tokenizer_tgt


@torch.no_grad()
def translate(model, tokenizer_src, tokenizer_tgt, args, device, sentence, max_len=128):
    """
    Translate an English sentence to Hindi.

    The decoder-only model maps source positions → target tokens.
    We feed the full source sequence and read off the predicted
    target tokens from the output logits.
    """
    model.eval()

    # Tokenize the English input
    src_tokens = tokenizer_src.encode(sentence).ids
    src_pad_id = tokenizer_src.token_to_id("[PAD]")

    # Truncate if needed
    src_tokens = src_tokens[:args.max_Seq_len]
    pad_len = args.max_Seq_len - len(src_tokens)

    # Pad to max_Seq_len
    src = torch.tensor(
        src_tokens + [src_pad_id] * pad_len,
        dtype=torch.long
    ).unsqueeze(0).to(device)  # (1, Seq_Len)

    # Forward pass
    output = model(src)  # (1, Seq_Len, tgt_vocab_size)

    # Get predicted token IDs (greedy decoding)
    pred_ids = output.argmax(dim=-1).squeeze(0).tolist()  # (Seq_Len,)

    # Decode only up to the length of the source (ignore padding positions)
    tgt_eos_id = tokenizer_tgt.token_to_id("[EOS]")
    tgt_pad_id = tokenizer_tgt.token_to_id("[PAD]")

    # Collect tokens until EOS or PAD
    result_ids = []
    for token_id in pred_ids:
        if token_id == tgt_eos_id or token_id == tgt_pad_id:
            break
        result_ids.append(token_id)

    # Decode token IDs back to Hindi text
    translated_text = tokenizer_tgt.decode(result_ids)
    return translated_text


def inference():
    # ====== Config ======
    CHECKPOINT_PATH = "../checkpoints/model_best.pt"
    TOKENIZER_DIR = "../tokenizers/"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ====== Load Tokenizers ======
    tokenizer_src, tokenizer_tgt = load_tokenizers(TOKENIZER_DIR)

    # ====== Setup Model ======
    args = ModelArgs()
    args.device = device
    args.src_vocab_size = tokenizer_src.get_vocab_size()
    args.tgt_vocab_size = tokenizer_tgt.get_vocab_size()

    print(f"Src Vocab Size: {args.src_vocab_size}")
    print(f"Tgt Vocab Size: {args.tgt_vocab_size}")
    print(args)

    # ====== Load Model ======
    model = Transformer(args).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False))
    model.eval()
    print(f"\nModel loaded from: {CHECKPOINT_PATH}")

    # ====== Translate ======
    test_sentences = [
        "Hello, how are you?",
        "What is your name?",
        "I love India.",
        "The weather is very nice today.",
        "Where is the train station?",
    ]

    print(f"\n{'='*60}")
    print("  ENGLISH → HINDI TRANSLATION")
    print(f"{'='*60}\n")

    for sentence in test_sentences:
        translation = translate(model, tokenizer_src, tokenizer_tgt, args, device, sentence)
        print(f"  EN: {sentence}")
        print(f"  HI: {translation}")
        print(f"  {'-'*56}")

    # ====== Interactive Mode ======
    print(f"\n{'='*60}")
    print("  INTERACTIVE MODE (type 'quit' to exit)")
    print(f"{'='*60}\n")

    while True:
        sentence = input("  EN: ").strip()
        if sentence.lower() in ('quit', 'exit', 'q'):
            break
        if not sentence:
            continue
        translation = translate(model, tokenizer_src, tokenizer_tgt, args, device, sentence)
        print(f"  HI: {translation}")
        print()


if __name__ == "__main__":
    inference()