# %% Step 1
import matplotlib.pyplot as plt

def plot_boundary(ax, predict, title):
    xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    ax.contourf(xx, yy, predict(grid).reshape(xx.shape), levels=[0, 0.5, 1], alpha=0.3)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=4)
    ax.set_title(title)

numpy_net = lambda g: sigmoid(np.tanh(g @ W1 + b1) @ W2 + b2).ravel()
def torch_net(g, m=model):
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(g, dtype=torch.float32))).squeeze(1).numpy()

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
plot_boundary(axes[0], lambda g: lr.predict_proba(g)[:, 1], f"Logistic regression ({lr.score(X_test, y_test):.3f})")
plot_boundary(axes[1], numpy_net, "NumPy network")
plot_boundary(axes[2], torch_net, "PyTorch network")
plt.tight_layout(); plt.show()

# %% Step 2
def train_torch(X_tr, y_tr, hidden, epochs=500, seed=0, X_val=None, y_val=None, patience=None):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(X_tr.shape[1], hidden), nn.Tanh(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    Xt_, yt_ = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)
    best, best_state, since = np.inf, None, 0
    for epoch in range(epochs):
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(m(Xt_).squeeze(1), yt_).backward()
        opt.step()
        if patience is not None:                               # early stopping on validation loss
            with torch.no_grad():
                vl = nn.functional.binary_cross_entropy_with_logits(
                    m(torch.tensor(X_val, dtype=torch.float32)).squeeze(1),
                    torch.tensor(y_val, dtype=torch.float32)).item()
            if vl < best - 1e-4:
                best, best_state, since = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                since += 1
                if since >= patience:
                    m.load_state_dict(best_state)
                    return m, epoch + 1
    return m, epochs

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, h in zip(axes, [2, 4, 64]):
    m, _ = train_torch(X_train, y_train, h)
    acc = ((torch_net(X_test, m) > 0.5) == y_test).mean()
    print(f"{h:>2} hidden units: test accuracy {acc:.3f}")
    plot_boundary(ax, lambda g: torch_net(g, m), f"{h} hidden units ({acc:.3f})")
plt.tight_layout(); plt.show()

# %% Note
# With the same 500 Adam steps, two and four hidden units do little better than logistic regression (0.853 and 0.862):
# their boundaries bend only slightly. Sixty-four units reach 0.940, about the same as the lesson's 16-unit network,
# so beyond the capacity the problem needs, extra units add nothing here (and training length matters too: small
# networks may need more steps to use what capacity they have).

# %% Step 3
Xn, yn = make_moons(n_samples=300, noise=0.45, random_state=1)            # small and noisy
Xn_tr, Xn_rest, yn_tr, yn_rest = train_test_split(Xn, yn, test_size=0.5, random_state=0)
Xn_val, Xn_te, yn_val, yn_te = train_test_split(Xn_rest, yn_rest, test_size=0.5, random_state=0)
for label, kwargs in [("64 units, 3000 epochs", {}),
                      ("64 units, early stopping", dict(X_val=Xn_val, y_val=yn_val, patience=50))]:
    m, epochs_run = train_torch(Xn_tr, yn_tr, 64, epochs=3000, **kwargs)
    tr_acc = ((torch_net(Xn_tr, m) > 0.5) == yn_tr).mean()
    te_acc = ((torch_net(Xn_te, m) > 0.5) == yn_te).mean()
    print(f"{label:<26} stopped after {epochs_run:>4} epochs: train {tr_acc:.3f}, test {te_acc:.3f}")
print(f"{'Logistic regression':<26} test {LogisticRegression().fit(Xn_tr, yn_tr).score(Xn_te, yn_te):.3f}")

# %% Note
# With 150 noisy training points, the 64-unit network trained for 3,000 epochs reaches 100% training accuracy: it has
# memorised the noise. Early stopping halted training after about 70 epochs, at 80% training accuracy. On this small
# test set (75 points) the two score similarly (0.747 and 0.733, a difference of about one point), and neither beats
# logistic regression (0.733), because with so much noise there is little non-linear structure to learn. The gap
# between training and test accuracy is the warning sign of overfitting; early stopping removes it.

# %% Step 4
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

# The synthetic diabetes-risk cohort from the logistic-regression lesson (same code and seed)
rng5 = np.random.default_rng(5)
n5 = 5000
dr = pd.DataFrame({"age": rng5.integers(30, 75, n5), "bmi": rng5.normal(27, 4.5, n5).round(1),
                   "fasting_glucose": rng5.normal(100, 10, n5).round(),
                   "family_history": rng5.binomial(1, 0.3, n5), "hypertension": rng5.binomial(1, 0.35, n5)})
logit5 = (-17.5 + 0.03 * dr.age + 0.10 * dr.bmi + 0.11 * dr.fasting_glucose + 0.6 * dr.family_history + 0.4 * dr.hypertension)
dr["diabetes_5y"] = rng5.binomial(1, 1 / (1 + np.exp(-logit5)))
predictors = ["age", "bmi", "fasting_glucose", "family_history", "hypertension"]
tr, te = train_test_split(dr, test_size=0.3, random_state=0, stratify=dr.diabetes_5y)
scaler = StandardScaler().fit(tr[predictors])
Xd_tr, Xd_te = scaler.transform(tr[predictors]), scaler.transform(te[predictors])

lr_d = LogisticRegression(penalty=None).fit(Xd_tr, tr.diabetes_5y)
Xd_fit, Xd_val, yd_fit, yd_val = train_test_split(Xd_tr, tr.diabetes_5y.to_numpy(), test_size=0.2, random_state=0)
net_d, _ = train_torch(Xd_fit, yd_fit, 16, epochs=2000, X_val=Xd_val, y_val=yd_val, patience=50)
results = {"Logistic regression": lr_d.predict_proba(Xd_te)[:, 1], "Neural network (16 units)": torch_net(Xd_te, net_d)}
fig, ax = plt.subplots(figsize=(4.5, 4.5))
for name, p in results.items():
    print(f"{name:<26} AUROC {roc_auc_score(te.diabetes_5y, p):.3f}  Brier {brier_score_loss(te.diabetes_5y, p):.4f}")
    obs, pred = calibration_curve(te.diabetes_5y, p, n_bins=10, strategy="quantile")
    ax.plot(pred, obs, "o-", label=name)
ax.plot([0, 0.8], [0, 0.8], ":", color="grey"); ax.set(xlabel="Predicted", ylabel="Observed"); ax.legend()
plt.show()

# %% Note
# The risk data were generated by a logistic model, so the network has nothing to find that logistic regression
# misses, and the two perform almost identically. Neural networks earn their place with images, signals and text,
# where useful features are hard to write down by hand, not with a few well-chosen tabular predictors.
