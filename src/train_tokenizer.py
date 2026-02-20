import sentencepiece as spm
from dataset import Shakes_Pear_Dataset
from model_args import ModelArgs

args=ModelArgs()
data=Shakes_Pear_Dataset(seq_len=args.max_Seq_len,tokenizer=None,mode="train")

# Train a BPE tokenizer on the Shakespeare dataset
spm.SentencePieceTrainer.Train(
    input=data.train_tokens,
    model_prefix='../tokenizer',       # outputs ../tokenizer.model and ../tokenizer.vocab
    vocab_size=2000,                    # small vocab for efficient training
    model_type='bpe',
    character_coverage=1.0,
    pad_id=3,
)

print("Tokenizer trained successfully!")
print("Files created: ../tokenizer.model, ../tokenizer.vocab")
