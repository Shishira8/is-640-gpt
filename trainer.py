import torch

class Trainer:
    def __init__(self, model, data_obj, max_iters=5000, eval_interval=500, eval_iters=200, batch_size=64, learning_rate=3e-4, device='cpu'):
        # Move model to the appropriate device (CPU or GPU)
        self.model = model.to(device)
        self.data = data_obj
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.batch_size = batch_size
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.device = device

        # Split data into training and validation sets
        self.train_data, self.val_data = self.data.get_splits()
      
    @torch.no_grad()
    def compute_loss(self):
        loss_dict = {}
        self.model.eval()
        for split in ['train', 'val']:
            loss_values = torch.zeros(self.eval_iters, device=self.device)
            for idx in range(self.eval_iters):
                X_batch, Y_batch = self.data.get_batch(split, self.batch_size, self.train_data, self.val_data)
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                _, loss = self.model(X_batch, Y_batch)
                loss_values[idx] = loss.item()
            loss_dict[split] = loss_values.mean()
        self.model.train()
        return loss_dict

    def run_training(self):
        for iteration in range(self.max_iters):
            if iteration % self.eval_interval == 0 or iteration == self.max_iters - 1:
                losses = self.compute_loss()
                print(f"Iteration {iteration}: Training Loss {losses['train']:.4f}, Validation Loss {losses['val']:.4f}")

            # Sample a batch of training data
            X_batch, Y_batch = self.data.get_batch('train', self.batch_size, self.train_data, self.val_data)
            X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

            # Perform forward pass, calculate loss, and backpropagate
            _, loss = self.model(X_batch, Y_batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        # After training is complete, evaluate final losses
        final_losses = self.compute_loss()
        print(f"Final Training Loss: {final_losses['train']:.4f}, Final Validation Loss: {final_losses['val']:.4f}")
