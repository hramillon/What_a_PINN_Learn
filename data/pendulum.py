import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

class NaiveNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, t):
        return self.net(t)

model = NaiveNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

df = pd.read_csv('data/data_10000.csv')

# Entrées (3 colonnes) et Sorties (2 colonnes)
X = torch.tensor(df[['theta_t', 'omega_t', 'u_t']].values, dtype=torch.float32)
Y = torch.tensor(df[['theta_next', 'omega_next']].values, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    pred = model(X_train)
    loss = criterion(pred, y_train)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'models/naiveNN_model.pth')