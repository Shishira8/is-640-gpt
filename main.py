import torch
from trainer import Trainer
from text_data import TextData  # Updated import to reflect the new class name
from model import GPTLanguageModel

# Constants
RANDOM_SEED = 1337
TRAIN_ITERATIONS = 1000
WORD_COUNT = 200
DATA_FILE = "input.txt"
OUTPUT_FILE = "generated_output.txt"

def main():
    torch.manual_seed(RANDOM_SEED)

    # Initialize the data, model, and trainer
    data = TextData(DATA_FILE)  # Updated class name
    model = GPTLanguageModel(data.vocab_size)
    trainer = Trainer(data, model)
    trainer.train(TRAIN_ITERATIONS)
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, WORD_COUNT)[0].tolist()

    output_text = data.decode_text(generated) 
    print(output_text)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(output_text)

if __name__ == "__main__":
    main()
