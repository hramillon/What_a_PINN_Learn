import numpy as np
import pandas as pd

np.random.seed(42)

g, m, L = 10.0, 1.0, 1.0
dt = 0.05
max_speed = 8.0
steps_per_traj = 50

def step_pendulum(theta, omega, u):
    # Ta mise à jour exacte
    new_omega = omega + ((3 * g) / (2 * L) * np.sin(theta) + (3 * u) / (m * L**2)) * dt
    new_omega = np.clip(new_omega, -max_speed, max_speed)
    
    new_theta = theta + new_omega * dt
    # On garde l'angle entre -pi et pi pour éviter que la valeur n'explose
    new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
    
    return new_theta, new_omega

for size in [100, 1000, 10000]:
    data = []
    traj_id = 0
    while len(data) < size:
        theta = np.random.uniform(-np.pi, np.pi)
        omega = np.random.uniform(-max_speed, max_speed)
        traj_id += 1
        
        for _ in range(steps_per_traj):
            if len(data) >= size: break
            
            u = np.random.uniform(-2.0, 2.0)
            next_theta, next_omega = step_pendulum(theta, omega, u)
            
            data.append([traj_id, theta, omega, u, next_theta, next_omega])
            theta, omega = next_theta, next_omega
            
    df = pd.DataFrame(data, columns=['trajectory_id','theta_t', 'omega_t', 'u_t', 'theta_next', 'omega_next'])
    df.to_csv(f'data_{size}.csv', index=False)