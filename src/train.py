import torch
from torch import nn
from torch.nn import functional as F
from model_args import ModelArgs
from model import Transformer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from dataset import Shakes_Pear_Dataset
import sentencepiece as spm

def train():
    torch.manual_seed(42)

    # ====== Hyperparameters ======
    BATCH_SIZE = 72
    EPOCHS = 15
    LR = 3e-4
    SAVE_DIR = "../checkpoints_sentencepiece_2"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ====== Device ======
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ====== Dataset & DataLoader ======
    args = ModelArgs()
    args.device = device

    # Tokenizer
    tokenizer=spm.SentencePieceProcessor()
    tokenizer.load("../tokenizer.model")

    train_dataset = Shakes_Pear_Dataset(args.max_Seq_len,tokenizer,"train")
    args.vocab_size = train_dataset.get_vocab_size()
    print(f"Vocab Size: {args.vocab_size}")
    print(f"Train Dataset Size: {len(train_dataset)} samples")
    
    print(args)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False,
    )
    
    # ====== Model ======
    model = Transformer(args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")


    # ====== Optimizer & Loss ======
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # ====== Learning Rate Scheduler (Cosine Annealing) ======
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    start_epoch = 0
    # checkpoint_path = "../checkpoints_sentencepiece/checkpoint_epoch_9.pt"

    # checkpoint = torch.load(checkpoint_path, map_location=device)

    # model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # start_epoch = checkpoint["epoch"]

    # print(f"Model resumed successfully from epoch {start_epoch}")



    # ====== Training Loop ======
    model.train()

    for epoch in range(start_epoch,EPOCHS):
        total_loss = 0
        num_batches = 0
        batch_losses = []  # Collect all per-batch losses for histogram

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

        for batch in progress_bar:
            src = batch["src"].to(device)       # (B, Seq_Len)
            target = batch["target"].to(device) # (B, Seq_Len)

            # Forward pass
            # output: (B, Seq_Len, vocab_size)
            output = model(src)

            # Reshape for cross entropy: (B * Seq_Len, vocab_size) vs (B * Seq_Len)
            loss = criterion(
                output.view(-1, args.vocab_size),
                target.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            batch_losses.append(loss.item())

            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # Step the scheduler
        scheduler.step()

        epoch_avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1}/{EPOCHS} — Avg Loss: {epoch_avg_loss:.4f} — LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint every epoch
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": epoch_avg_loss,
            "args": args,
        }
        torch.save(checkpoint, os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt"))
        print(f"Checkpoint saved: checkpoint_epoch_{epoch+1}.pt")

        # Save loss histogram
        plt.figure(figsize=(10, 6))
        plt.hist(batch_losses, bins=50, color='steelblue', edgecolor='black', alpha=0.8)
        plt.title(f"Loss Distribution — Epoch {epoch+1} (Avg: {epoch_avg_loss:.4f})")
        plt.xlabel("Loss")
        plt.ylabel("Frequency")
        plt.axvline(epoch_avg_loss, color='red', linestyle='--', label=f'Mean: {epoch_avg_loss:.4f}')
        plt.legend()
        plt.tight_layout()
        hist_path = os.path.join(SAVE_DIR, f"loss_hist_epoch_{epoch+1}.png")
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"Loss histogram saved: {hist_path}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model_final.pt"))
    print("\nTraining complete! Final model saved.")


if __name__ == "__main__":
    train()