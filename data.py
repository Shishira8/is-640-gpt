import torch

class Data:
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.text = self.text.replace('\n', ' ').replace('\r', '').strip()
        self.chars = sorted(list(set(self.text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # Split the text data into train and validation sets
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        self.n_train = int(0.9 * len(self.data))  # 90% for training
        self.train_data = self.data[:self.n_train]
        self.val_data = self.data[self.n_train:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data
