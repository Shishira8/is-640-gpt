import torch
from trainer import Trainer
from data import Data
from model import GPTLanguageModel

RANDOM_SEED = 2468
TRAIN_ITERATIONS = 100
WORD_COUNT = 200
DATA_FILE = "input.txt"
BATCH_SIZE = 32  # Added batch size

def main():
    torch.manual_seed(RANDOM_SEED)
    data = Data(DATA_FILE)
    model = GPTLanguageModel(data.vocab_size, batch_size=BATCH_SIZE)
    trainer = Trainer(data, model, batch_size=BATCH_SIZE)
    trainer.train(TRAIN_ITERATIONS)
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, WORD_COUNT)[0].tolist()
    print(data.decode(generated))
    # To-do: Write output to a file
    #with open("output.txt", "w", encoding="utf-8") as f:
    #    f.write(data.decode(generated))

if __name__ == "__main__":
    main()
