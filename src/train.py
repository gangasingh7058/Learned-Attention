import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from pathlib import Path

from model_args import ModelArgs
from model import Transformer
from dataset import BilingualDataset

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


# ====================================================================
#                     TOKENIZER HELPERS
# ====================================================================

def get_all_sentences(ds, lang):
    """Yield all sentences for a given language from the dataset."""
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(ds, lang, tokenizer_dir):
    """Load an existing tokenizer or train a new one for a specific language."""
    tokenizer_path = Path(tokenizer_dir) / f"tokenizer_{lang}.json"
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
            vocab_size=5000,
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer trained and saved: {tokenizer_path}")
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"Tokenizer loaded from: {tokenizer_path}")
    return tokenizer


# ====================================================================
#                        TRAINING
# ====================================================================

def train():
    torch.manual_seed(42)

    # ====== Hyperparameters ======
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 3e-4
    SAVE_DIR = "../checkpoints/"
    TOKENIZER_DIR = "../tokenizers/"

    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"LR: {LR}")
    print(f"SAVE_DIR: {SAVE_DIR}")
    print(f"TOKENIZER_DIR: {TOKENIZER_DIR}")
    # Dataset config
    DATASET_NAME = "Helsinki-NLP/opus-100"
    SRC_LANG = "en"
    TGT_LANG = "hi"

    # Resume training (set to checkpoint path to resume, or None to train from scratch)
    RESUME_CHECKPOINT = "../checkpoints/checkpoint_epoch_7.pt"  # e.g., "../checkpoints/checkpoint_epoch_5.pt"

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ====== Device ======
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ====== Model Args ======
    args = ModelArgs()
    args.device = device

    # ====== Load Dataset ======
    print(f"\nLoading dataset: {DATASET_NAME} ({SRC_LANG}-{TGT_LANG})...")
    ds_raw = load_dataset(DATASET_NAME, f"{SRC_LANG}-{TGT_LANG}", split="train")

    # ====== Build / Load Separate Tokenizers ======
    tokenizer_src = get_or_build_tokenizer(ds_raw, SRC_LANG, TOKENIZER_DIR)
    tokenizer_tgt = get_or_build_tokenizer(ds_raw, TGT_LANG, TOKENIZER_DIR)

    args.src_vocab_size = tokenizer_src.get_vocab_size()
    args.tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    tgt_pad_id = tokenizer_tgt.token_to_id("[PAD]")

    # ====== Train / Val Split ======
    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_ds_raw, val_ds_raw = torch.utils.data.random_split(ds_raw, [train_size, val_size])

    train_dataset = BilingualDataset(
        train_ds_raw, tokenizer_src, tokenizer_tgt,
        SRC_LANG, TGT_LANG, args.max_Seq_len
    )
    val_dataset = BilingualDataset(
        val_ds_raw, tokenizer_src, tokenizer_tgt,
        SRC_LANG, TGT_LANG, args.max_Seq_len
    )

    print(f"\nSrc Vocab Size : {args.src_vocab_size}")
    print(f"Tgt Vocab Size : {args.tgt_vocab_size}")
    print(f"Train samples  : {len(train_dataset):,}")
    print(f"Val samples    : {len(val_dataset):,}")
    print(f"\n{args}\n")

    # ====== DataLoaders ======
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == 'cuda'),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == 'cuda'),
    )

    # ====== Model ======
    model = Transformer(args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters    : {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # ====== Optimizer & Loss ======
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id)

    # ====== Learning Rate Scheduler (Cosine Annealing) ======
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ====== Resume from Checkpoint (optional) ======
    start_epoch = 0
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resumed from checkpoint: {RESUME_CHECKPOINT} (epoch {start_epoch})")
    else:
        print("Training from scratch.")

    # ====== Training Loop ======
    best_val_loss = float('inf')

    for epoch in range(start_epoch, EPOCHS):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        total_loss = 0
        num_batches = 0
        batch_losses = []

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

        for batch in progress_bar:
            src = batch["src"].to(device)    # (B, Seq_Len) — English input
            tgt = batch["tgt"].to(device)    # (B, Seq_Len) — French label

            # Forward pass: English tokens in → French logits out
            output = model(src)  # (B, Seq_Len, tgt_vocab_size)

            # Reshape for cross entropy
            loss = criterion(
                output.view(-1, args.tgt_vocab_size),
                tgt.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            batch_losses.append(loss.item())

            avg_loss = total_loss / num_batches
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{avg_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        scheduler.step()

        epoch_avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1}/{EPOCHS} — Train Loss: {epoch_avg_loss:.4f} — LR: {optimizer.param_groups[0]['lr']:.2e}")

        # ── Validation ───────────────────────────────────────────────
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation", leave=False):
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                output = model(src)
                loss = criterion(output.view(-1, args.tgt_vocab_size), tgt.view(-1))
                val_loss += loss.item()
                val_batches += 1

        val_avg_loss = val_loss / val_batches
        print(f"Validation Loss : {val_avg_loss:.4f}")

        # ── Save Best Model ──────────────────────────────────────────
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model_best.pt"))
            print(f"New best model saved (val_loss: {val_avg_loss:.4f})")

        # ── Save Checkpoint ──────────────────────────────────────────
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": epoch_avg_loss,
            "val_loss": val_avg_loss,
            "args": args,
        }
        torch.save(checkpoint, os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt"))
        print(f"Checkpoint saved: checkpoint_epoch_{epoch+1}.pt")

        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # ── Loss Histogram ───────────────────────────────────────────
        plt.figure(figsize=(10, 6))
        plt.hist(batch_losses, bins=50, color='steelblue', edgecolor='black', alpha=0.8)
        plt.title(f"Loss Distribution — Epoch {epoch+1} (Train: {epoch_avg_loss:.4f} | Val: {val_avg_loss:.4f})")
        plt.xlabel("Loss")
        plt.ylabel("Frequency")
        plt.axvline(epoch_avg_loss, color='red', linestyle='--', label=f'Train Mean: {epoch_avg_loss:.4f}')
        plt.legend()
        plt.tight_layout()
        hist_path = os.path.join(SAVE_DIR, f"loss_hist_epoch_{epoch+1}.png")
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"Loss histogram saved: {hist_path}")

    # ====== Save Final Model ======
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model_final.pt"))
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"  Final model saved to: {SAVE_DIR}model_final.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()