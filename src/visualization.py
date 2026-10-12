import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Même architecture que celle du modèle entraîné
class NaiveNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), 
            nn.Tanh(),
            nn.Linear(64, 64), 
            nn.Tanh(),
            nn.Linear(64, 2)  # Prédit [theta_next, omega_next]
        )
    def forward(self, x):
        return self.net(x)

# 1. Chargement du modèle
model = NaiveNN()
model.load_state_dict(torch.load('models/naiveNN_model.pth'))
model.eval()

# 2. Paramètres initiaux
theta_t = np.random.uniform(-np.pi, np.pi)
omega_t = 0.0
u_t = 0.0  # Couple neutre

trajectoire = [theta_t]

# 3. Prédiction de la trajectoire
with torch.no_grad():
    for _ in range(150):
        # On passe les 3 variables requises en entrée
        entrees = torch.tensor([[theta_t, omega_t, u_t]], dtype=torch.float32)
        pred = model(entrees)
        
        theta_next = pred[0, 0].item()
        omega_next = pred[0, 1].item()
        
        # Modulo pour normaliser l'angle entre -pi et pi
        theta_t = (theta_next + np.pi) % (2 * np.pi) - np.pi
        omega_t = omega_next
        
        trajectoire.append(theta_t)

# 4. Configuration de l'affichage
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.grid(True)

# Ligne principale (le pendule) et traînée (les positions passées) en rouge
line, = ax.plot([], [], 'o-', lw=3, color='red', markersize=10)
trail, = ax.plot([], [], '-', lw=1.5, color='red', alpha=0.3)

# 5. Fonction d'animation
def animate(i):
    # Les équations d'origine du pendule inversé font que l'angle 0 est en haut.
    # Pour qu'il pende normalement, on utilise -cos pour l'axe Y.
    x = np.sin(trajectoire[i])
    y = -np.cos(trajectoire[i])
    line.set_data([0, x], [0, y])
    
    # Traînée historique (jusqu'aux 20 frames précédentes)
    start = max(0, i - 20)
    xs = np.sin(trajectoire[start:i+1])
    ys = -np.cos(trajectoire[start:i+1])
    trail.set_data(xs, ys)
    
    return line, trail

ani = animation.FuncAnimation(fig, animate, frames=len(trajectoire), interval=50, blit=True)
plt.show()