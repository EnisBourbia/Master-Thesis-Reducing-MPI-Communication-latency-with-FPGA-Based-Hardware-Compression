import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
# Setup MPI
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group("mpi")
rank = dist.get_rank()
world_size = dist.get_world_size()

def load_data():
    x = torch.load("./data/wikitext2_train_tensor.pt", weights_only=False)
    y = x.clone()  # simple next-token prediction
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(50258, 128)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(256, 50258)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        return self.fc(x)

def main():
    model = ToyLM()
    model = DDP(model)
    loader = load_data()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1):
        for batch in loader:
            x, y = batch
            y = y.view(-1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, 50258), y)
            loss.backward()
            optimizer.step()
            print(f"Rank {rank} - Loss: {loss.item():.4f}")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()