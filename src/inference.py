import torch
from model import Transformer
from model_args import ModelArgs
import sentencepiece as spm

def inference():
    MAX_GEN = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load("../tokenizer.model")

    # Setup model args
    args = ModelArgs()
    args.device = device
    args.vocab_size = tokenizer.vocab_size()
    print(f"Vocab Size: {args.vocab_size}")
    print(args)

    # Load model
    model = Transformer(args).to(device)
    model.load_state_dict(torch.load("../checkpoints_sentencepiece/model_final.pt", map_location=device))
    model.eval()

    # Encode prompt
    prompt = "To be or not to be, that is"
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    eos_token = tokenizer.eos_id()

    # Generate
    generated_tokens = tokens.clone()
    with torch.no_grad():
        for _ in range(MAX_GEN):
            # Only feed last max_Seq_len tokens to avoid exceeding model's max length
            input_tokens = generated_tokens[:, -args.max_Seq_len:]
            output = model(input_tokens)
            output = output[:, -1, :]
            output = torch.softmax(output, dim=-1)
            next_token = torch.multinomial(output, num_samples=1)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
            if next_token.item() == eos_token:
                break

    # Decode
    output = tokenizer.decode(generated_tokens.squeeze().tolist())
    return output

if __name__ == "__main__":
    output = inference()

    print("\n\n")
    print("-" * 60)
    print(output)
    print("-" * 60)
    print("\n\n")