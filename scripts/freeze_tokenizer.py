from transformers import GPT2TokenizerFast
from pathlib import Path

TOKENIZER_PATH = Path(__file__).parent.parent / "nic" / "tokenizer"

def main():
    print("Downloading GPT-2 tokenizer from HuggingFace...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    
    TOKENIZER_PATH.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(TOKENIZER_PATH)
    
    print(f"Tokenizer saved to {TOKENIZER_PATH}")
    print("Files written:")
    for f in sorted(TOKENIZER_PATH.iterdir()):
        print(f"  {f.name}")

if __name__ == "__main__":
    main()