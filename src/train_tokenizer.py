import sentencepiece as spm

# Train a BPE tokenizer on the Shakespeare dataset
spm.SentencePieceTrainer.Train(
    input='../Datasets/Shakespear_dataset/Shakes_Pear.txt',
    model_prefix='../tokenizer',       # outputs ../tokenizer.model and ../tokenizer.vocab
    vocab_size=2000,                    # small vocab for efficient training
    model_type='bpe',
    character_coverage=1.0,
    pad_id=3,
)

print("Tokenizer trained successfully!")
print("Files created: ../tokenizer.model, ../tokenizer.vocab")
