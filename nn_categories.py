import torch
import pandas as pd
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


X = torch.load('X.pt')
y = torch.load('y.pt')
X = F.normalize(X, p=1.0, dim=-1)
print(X)

#6
class CCC(nn.Module):
    def __init__(self,  in_, out_, neurons=100):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(in_features=in_, out_features=neurons),
            nn.ReLU(),
            nn.Linear(in_features=neurons, out_features=neurons),
            nn.ReLU(),
            nn.Linear(in_features=neurons, out_features=neurons),
            nn.ReLU(),
            nn.Linear(in_features=neurons, out_features=neurons),
            nn.ReLU(),
            nn.Linear(in_features=neurons, out_features=neurons),
            nn.ReLU(),
            nn.Linear(in_features=neurons, out_features=out_)
        )
    def forward(self, x):
        return self.stack(x)

model0 = CCC(in_=16, out_=18, neurons=300)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model0.parameters(), lr=0.001)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

train_dataset = TensorDataset(X_train, y_train)
batch_size = 320  # Set your batch size here
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

epochs = 1
losses = []
for epoch in range(epochs):
    torch.manual_seed(100)
    model0.train()
    # y_preds = model0(X_train)
    # print(y_train)
    # print(y_preds)
    # loss = loss_fn(y_preds, y_train)
    # optimizer.zero_grad()
    # loss.backward()
    # torch.nn.utils.clip_grad_norm_(model0.parameters(), max_norm=1.0)
    # optimizer.step()
    # losses.append(loss.item())
    # print(f'epoch {epoch + 1}/{epochs}, loss: {loss.item()}')
    epoch_loss = 0  # Initialize epoch loss
    for batch_inputs, batch_targets in train_loader:
        y_preds = model0(batch_inputs)
        loss = loss_fn(y_preds, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model0.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f'epoch {epoch + 1}/{epochs}, loss: {avg_loss}')

plt.plot(range(1, epochs + 1), losses, label='training loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('training loss')
plt.legend()
plt.show()

y_probs = F.softmax(y_preds, dim=1)
print(y_train.sum(dim=-2))
print(y_probs.sum(dim=-2))

#TESTING LOOP
model0.eval() #tryb testowania
test_preds = model0(X_test)
test_loss = loss_fn(test_preds, y_test)
print(test_loss)

test_probs = F.softmax(test_preds, dim=1)
print(y_test.sum(dim=-2))
print(test_probs.sum(dim=-2))

from pathlib import Path

model_path = Path('models')
model_path.mkdir(parents=True, exist_ok=True)
model_name = 'model0.pth'
model_save_path = model_path / model_name
torch.save(obj=model0.state_dict(),f=model_save_path)

loaded = CCC(in_=16, out_=18, neurons=300)
loaded.load_state_dict(torch.load(f=model_save_path))