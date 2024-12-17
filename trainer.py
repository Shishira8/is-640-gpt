import torch
from torch.optim.lr_scheduler import StepLR

class Trainer:
    def __init__(self, data, model, learning_rate=3e-4, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.data = data
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.device = device
        self.batch_size = batch_size

        # Using the split data from Data class
        self.train_data = data.get_train_data()
        self.val_data = data.get_val_data()

    def train(self, iterations):
        for iter in range(iterations):
            # Get batch from training data
            x, y = self.get_batch(self.train_data)
            logits, loss = self.model(x, y)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Print loss and run validation every 10 iterations
            if iter % 10 == 0 or iter == iterations - 1:
                print(f"Iteration {iter}, Loss: {loss.item()}")
                self.validate()

    def get_batch(self, data):
        # Generate a batch of data
        ix = torch.randint(len(data) - self.model.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.model.block_size] for i in ix]).to(self.device)
        y = torch.stack([data[i + 1:i + self.model.block_size + 1] for i in ix]).to(self.device)
        return x, y

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            # Use the validation data for evaluation
            x, y = self.get_batch(self.val_data)
            logits, loss = self.model(x, y)
            val_loss += loss.item()
        print(f"Validation Loss: {val_loss}")
        self.model.train()
