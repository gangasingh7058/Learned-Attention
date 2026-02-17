import torch
from model import Transformer
from model_args import ModelArgs
from dataset import Shakes_Pear_Dataset
from torch.utils.data import DataLoader
import tiktoken

def inference():
    BATCH_SIZE=1
    args=ModelArgs()
    MAX_GEN=100
    device='cuda' if torch.cuda.is_available() else 'cpu'

    dataset=Shakes_Pear_Dataset(args.max_Seq_len)
    args.vocab_size = dataset.get_vocab_size()
    print(f"Vocab Size: {args.vocab_size}")
    print(f"Dataset Size: {len(dataset)} samples")
    
    print(args)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False,
    )

    model=Transformer(args).to(device)
    model.load_state_dict(torch.load("../checkpoints/model_final.pt"))
    model.eval()
    
    prompt = "To be or not to be, that is"
    tokenizer=tiktoken.get_encoding("gpt2")
    tokens=tokenizer.encode(prompt)
    tokens=torch.tensor(tokens,dtype=torch.long).unsqueeze(0).to(device)
    eos_token=tokenizer.eot_token

    generated_tokens=tokens.clone()
    with torch.no_grad():
        for _ in range(MAX_GEN):
            output=model(generated_tokens)
            output=output[:, -1, :]
            output=torch.softmax(output,dim=-1)
            next_token=torch.multinomial(output,num_samples=1)
            generated_tokens=torch.cat([generated_tokens,next_token],dim=-1)
            if next_token.item() == eos_token:
                break
    output = tokenizer.decode(generated_tokens.squeeze().tolist())
    return output

if __name__ == "__main__":
    output=inference()

    print("\n\n\n\n")
    print("-"*60)
    print(output)
    print("-"*60)
    print("\n\n\n\n")