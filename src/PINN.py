import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split


class PINN(nn.Module):
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

model = PINN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()


def physics_loss(X, Y_pred):
    # X contient [theta_t, omega_t, u_t]
    theta, omega, u_control = X[:, 0], X[:, 1], X[:, 2]
    
    # Y_pred contient les prédictions [theta_next, omega_next]
    theta_next_pred, omega_next_pred = Y_pred[:, 0], Y_pred[:, 1]
    
    g, m, L, dt = 10.0, 1.0, 1.0, 0.05
    
    # Calcul de l'état suivant selon la physique (sans np.clip/modulo pour rester différentiable)
    omega_next_phys = omega + ((3 * g) / (2 * L) * torch.sin(theta) + (3 * u_control) / (m * L**2)) * dt
    theta_next_phys = theta + omega_next_phys * dt
    
    # Résidu f : écart entre la prédiction du réseau et la loi physique
    f_theta = theta_next_pred - theta_next_phys
    f_omega = omega_next_pred - omega_next_phys
    
    return torch.mean(f_theta**2 + f_omega**2)


df = pd.read_csv('data/data_10000.csv')

# Entrées (3 colonnes) et Sorties (2 colonnes)
X = torch.tensor(df[['theta_t', 'omega_t', 'u_t']].values, dtype=torch.float32)
Y = torch.tensor(df[['theta_next', 'omega_next']].values, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

for epoch in range(10000):
    model.train()
    optimizer.zero_grad()
    
    pred = model(X_train)
    mse_u = mse(pred, y_train)          
    mse_f = physics_loss(X_train, pred)

    loss = mse_u + mse_f
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/10000], Loss: {loss.item():.4f}')

# Evaluation du modèle

model.eval()
with torch.no_grad():
    test_pred = model(X_test)

    mse_test_u = mse(test_pred, y_test)          
    mse_test_f = physics_loss(X_test, test_pred)
    
    test_loss = mse_test_f + mse_test_u
    print(f'Test Loss: {test_loss.item():.4f}')

torch.save(model.state_dict(), 'models/PINN_model.pth')

## model 100 points 1000 epochs
#Epoch [1000/1000], Loss: 0.7772
#Test Loss: 0.9252
# Leger overfitting

## model 1000 points 
#Epoch [10000/10000], Loss: 0.5618
#Test Loss: 0.6508
# Leger overfitting + mouvements hasardeux mais plus variès que avec 100 points