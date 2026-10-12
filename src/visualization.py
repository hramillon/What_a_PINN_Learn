import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

df = pd.read_csv('data/data_1000.csv')
# On anime 150 frames (3 trajectoires complètes de 50 pas)
theta = df['theta_t'].values[:150]

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.grid()

line, = ax.plot([], [], 'o-', lw=3, color='blue', markersize=10)

def animate(i):
    # L'angle 0 correspond à la position HAUTE dans l'équation de Gym
    x = np.sin(theta[i])
    y = np.cos(theta[i])
    line.set_data([0, x], [0, y])
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(theta), interval=50, blit=True)
plt.show()