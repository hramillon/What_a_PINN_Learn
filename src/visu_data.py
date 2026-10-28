import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Lecture de l'argument en ligne de commande
parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, help="Nombre de trajectoires du dataset à visualiser")
args = parser.parse_args()

# Chargement des données
df = pd.read_csv('data/data_1000.csv')

# Sélection aléatoire de n trajectoires
ids_uniques = df['trajectory_id'].unique()
trajets_choisis = np.random.choice(ids_uniques, size=args.n, replace=False)

for tid in trajets_choisis:
    # Extraction des données réelles de la trajectoire
    df_traj = df[df['trajectory_id'] == tid]
    
    # On récupère tous les theta_t et on ajoute le tout dernier theta_next pour la fin
    trajectoire = df_traj['theta_t'].tolist() + [df_traj['theta_next'].iloc[-1]]
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True)
    ax.set_title(f"Dataset - Trajectoire ID : {tid}")

    # En bleu pour différencier des prédictions du modèle
    line, = ax.plot([], [], 'o-', lw=3, color='blue', markersize=10)
    trail, = ax.plot([], [], '-', lw=1.5, color='blue', alpha=0.3)

    def animate(i):
        x = np.sin(trajectoire[i])
        y = np.cos(trajectoire[i])
        line.set_data([0, x], [0, y])
        
        start = max(0, i - 20)
        xs = np.sin(trajectoire[start:i+1])
        ys = np.cos(trajectoire[start:i+1])
        trail.set_data(xs, ys)
        
        return line, trail

    # repeat=False pour passer à la suivante une fois terminée (nécessite de fermer la fenêtre)
    ani = animation.FuncAnimation(fig, animate, frames=len(trajectoire), interval=50, blit=True, repeat=False)
    plt.show()