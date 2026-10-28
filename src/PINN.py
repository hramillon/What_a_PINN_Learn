import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------- arguments
p = argparse.ArgumentParser()
p.add_argument('--n', type=int, default=100, help="taille du dataset (100, 1000, 10000)")
p.add_argument('--lam', type=float, default=1.0, help="poids de la loss physique (0 = réseau purement supervisé)")
p.add_argument('--ncol', type=int, default=2000, help="points de collocation tirés à chaque epoch")
p.add_argument('--epochs', type=int, default=1000)
args = p.parse_args()

torch.manual_seed(0)
np.random.seed(0)

# ------------------------------------------------- constantes (= générateur)
G, M, L, DT = 10.0, 1.0, 1.0, 0.05
WMAX, UMAX = 8.0, 2.0
PI = torch.pi


def wrap(x):
    """Ramène un angle dans [-pi, pi)."""
    return torch.remainder(x + PI, 2 * PI) - PI


# ------------------------------------------------------------------ modèle
class PINN(nn.Module):
    """Entrée  : X = [theta_t, omega_t, u_t]  (theta brut, en radians)
    Sortie : [dtheta/(WMAX*DT), omega_next/WMAX]  (valeurs normalisées ~[-1, 1])
    theta n'est jamais donné brut au réseau : on donne sin/cos, donc plus de saut en +-pi.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, X):
        th, om, u = X[:, 0], X[:, 1], X[:, 2]
        feat = torch.stack([torch.sin(th), torch.cos(th), om / WMAX, u / UMAX], dim=1)
        return self.net(feat)


def physics(X):
    """Prédiction exacte de l'équation (mêmes formules que le générateur, clip inclus),
    dans les mêmes unités normalisées que la sortie du réseau."""
    th, om, u = X[:, 0], X[:, 1], X[:, 2]
    om_n = om + ((3 * G) / (2 * L) * torch.sin(th) + (3 * u) / (M * L ** 2)) * DT
    om_n = torch.clamp(om_n, -WMAX, WMAX)
    dth = om_n * DT
    return torch.stack([dth / (WMAX * DT), om_n / WMAX], dim=1)


def target_from_data(X, Y):
    """Convertit (theta_next, omega_next) du CSV en cibles normalisées.
    wrap(theta_next - theta_t) supprime le saut de 2*pi dû à l'enroulement."""
    dth = wrap(Y[:, 0] - X[:, 0])
    return torch.stack([dth / (WMAX * DT), Y[:, 1] / WMAX], dim=1)


def decode(out):
    """Sortie normalisée -> (dtheta, omega_next) en unités physiques."""
    return out[:, 0] * WMAX * DT, out[:, 1] * WMAX


# ------------------------------------------------------------ données
df = pd.read_csv(f'data/data_{args.n}.csv')

# Split PAR TRAJECTOIRE (pas par ligne) pour éviter la fuite train/test
ids = df['trajectory_id'].unique()
ids = np.random.permutation(ids)
n_test = max(1, int(0.2 * len(ids)))
test_ids, train_ids = ids[:n_test], ids[n_test:]
df_train = df[df['trajectory_id'].isin(train_ids)]
df_test = df[df['trajectory_id'].isin(test_ids)]
print(f"{len(train_ids)} trajectoires train ({len(df_train)} lignes), "
      f"{len(test_ids)} trajectoires test ({len(df_test)} lignes)")

cols_X = ['theta_t', 'omega_t', 'u_t']
cols_Y = ['theta_next', 'omega_next']


def to_tensors(d):
    X = torch.tensor(d[cols_X].values, dtype=torch.float32)
    Y = torch.tensor(d[cols_Y].values, dtype=torch.float32)
    return X, target_from_data(X, Y)


X_train, T_train = to_tensors(df_train)
X_test, T_test = to_tensors(df_test)

# ------------------------------------------------------------ entraînement
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()

best_val, best_state = float('inf'), None

for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()

    # 1) perte de données : prédiction vs. données du CSV
    loss_data = mse(model(X_train), T_train)

    # 2) perte physique : points de collocation tirés au hasard, SANS labels.
    #    Le réseau doit satisfaire l'équation partout dans l'espace des états.
    Xc = torch.empty(args.ncol, 3)
    Xc[:, 0].uniform_(-np.pi, np.pi)
    Xc[:, 1].uniform_(-WMAX, WMAX)
    Xc[:, 2].uniform_(-UMAX, UMAX)
    loss_phys = mse(model(Xc), physics(Xc))

    loss = loss_data + args.lam * loss_phys
    loss.backward()
    optimizer.step()

    # validation (perte de données uniquement) + sauvegarde du meilleur modèle
    model.eval()
    with torch.no_grad():
        val = mse(model(X_test), T_test).item()
    if val < best_val:
        best_val = val
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1}/{args.epochs}  data={loss_data.item():.2e}  "
              f"phys={loss_phys.item():.2e}  val={val:.2e}")

model.load_state_dict(best_state)
model.eval()
print(f"Meilleure val loss : {best_val:.2e}")

# ------------------------------------------------ évaluation en rollout
# Le réseau est rejoué seul, pas à pas, avec la vraie séquence de u.
# On compare à la trajectoire du CSV : c'est le test qui reflète l'aspect visuel.
def rollout_errors(d):
    err_th, err_om = [], []
    for _, g in d.groupby('trajectory_id', sort=False):
        th = torch.tensor(g['theta_t'].iloc[0], dtype=torch.float32)
        om = torch.tensor(g['omega_t'].iloc[0], dtype=torch.float32)
        th_true = torch.tensor(g['theta_next'].values, dtype=torch.float32)
        om_true = torch.tensor(g['omega_next'].values, dtype=torch.float32)
        us = torch.tensor(g['u_t'].values, dtype=torch.float32)
        e_th, e_om = [], []
        with torch.no_grad():
            for k in range(len(g)):
                x = torch.stack([th, om, us[k]]).unsqueeze(0)
                dth, om_n = decode(model(x))
                th = wrap(th + dth[0])
                om = om_n[0]
                e_th.append(wrap(th - th_true[k]).abs().item())
                e_om.append((om - om_true[k]).abs().item())
        err_th.append(e_th)
        err_om.append(e_om)
    return err_th, err_om


err_th, err_om = rollout_errors(df_test)
for k in (1, 10, 25, 50):
    ok_th = [e[k - 1] for e in err_th if len(e) >= k]
    ok_om = [e[k - 1] for e in err_om if len(e) >= k]
    if ok_th:
        print(f"Rollout pas {k:2d} : erreur |theta| = {np.mean(ok_th):.3f} rad, "
              f"erreur |omega| = {np.mean(ok_om):.3f} rad/s  ({len(ok_th)} traj.)")

os.makedirs('models', exist_ok=True)
torch.save(best_state, f'models/PINN_model_n{args.n}_lam{args.lam}.pth')