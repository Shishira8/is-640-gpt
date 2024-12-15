import torch

class TextData:
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.text_content = file.read()
        self.unique_chars = sorted(list(set(self.text_content)))
        self.vocab_size = len(self.unique_chars)
        self.char_to_index = {char: idx for idx, char in enumerate(self.unique_chars)}
        self.index_to_char = {idx: char for idx, char in enumerate(self.unique_chars)}
        self.tensor_data = torch.tensor(self._encode_text(self.text_content), dtype=torch.long)

    def _encode_text(self, text):
        return [self.char_to_index[char] for char in text]

    def decode_text(self, indices):
        return ''.join([self.index_to_char[idx] for idx in indices])

    def get_train_test_split(self, ratio=0.9):
        split_point = int(ratio * len(self.tensor_data))
        return self.tensor_data[:split_point], self.tensor_data[split_point:]
