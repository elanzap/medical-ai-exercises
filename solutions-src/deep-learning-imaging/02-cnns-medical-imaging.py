# %% Step 1
print(f"CNN {accuracy(cnn, X_test, y_test):.3f}; pixel logistic regression "
      f"{flat.score(X_test.reshape(len(X_test), -1).numpy(), y_test.numpy()):.3f}")
print("marker rate   same hospital   new hospital")
for rate in [0.5, 0.7, 0.9, 1.0]:
    Xd, yd = make_images(4000, seed=3, marker_with_lesion=rate)
    torch.manual_seed(0)
    m = train(SmallCNN(), Xd, yd)
    Xs, ys = make_images(1000, seed=4, marker_with_lesion=rate)
    print(f"{rate:>11.1f}   {accuracy(m, Xs, ys):>13.3f}   {accuracy(m, X_new, y_new):>12.3f}")

# %% Note
# The more reliably the marker accompanies lesions, the better the model looks at its own hospital and the worse it
# does where the marker never appears: at a rate of 1.0, the marker alone predicts the label perfectly.

# %% Step 2
def mask_corner(X):
    X = X.clone()
    X[:, :, :3, :3] = 0.0                     # remove the region where the marker appears
    return X

torch.manual_seed(0)
masked_cnn = train(SmallCNN(), mask_corner(X_dev), y_dev)
print(f"Shortcut model, new hospital:        {accuracy(shortcut_cnn, X_new, y_new):.3f}")
print(f"Corner-masked model, new hospital:   {accuracy(masked_cnn, mask_corner(X_new), y_new):.3f}")
print(f"Model trained without markers:       {accuracy(cnn, X_new, y_new):.3f}")

# %% Note
# Masking the corner removes the shortcut, so the network has to learn the lesion itself, and new-hospital accuracy
# recovers to the level of a model trained on clean images. In real images the shortcut is rarely so easy to locate
# (text labels, positioning, image processing), which is why external testing matters more than any single fix.

# %% Step 3
from medmnist import PneumoniaMNIST
from sklearn.metrics import confusion_matrix, roc_auc_score

def load_medmnist(cls, split):
    d = cls(split=split, download=True, root="data", size=28)
    X = torch.tensor(d.imgs, dtype=torch.float32).unsqueeze(1) / 255.0
    X = (X - 0.5) / 0.25                                   # roughly standardise intensities
    return X, torch.tensor(d.labels.ravel(), dtype=torch.long)

Xp_tr, yp_tr = load_medmnist(PneumoniaMNIST, "train")
Xp_te, yp_te = load_medmnist(PneumoniaMNIST, "test")
print(f"PneumoniaMNIST: {len(Xp_tr)} training and {len(Xp_te)} test images; pneumonia in "
      f"{yp_tr.float().mean():.0%} of training and {yp_te.float().mean():.0%} of test images")

torch.manual_seed(0)
pneu = train(SmallCNN(), Xp_tr, yp_tr, epochs=8)

def report(model, X, y, label):
    with torch.no_grad():
        p = model(X).softmax(1)[:, 1].numpy()
    tn, fp, fn, tp = confusion_matrix(y.numpy(), p >= 0.5).ravel()
    print(f"{label:<28} AUROC {roc_auc_score(y.numpy(), p):.3f}  sensitivity {tp / (tp + fn):.3f}  "
          f"specificity {tn / (tn + fp):.3f}")

report(pneu, Xp_te, yp_te, "SmallCNN")

# %% Note
# PneumoniaMNIST is a 28x28 version of a public paediatric chest X-ray dataset (MedMNIST, CC BY 4.0). Typical of
# this dataset, sensitivity is very high and specificity much lower at the default threshold (0.995 and 0.538 here),
# and the test set has a different class balance from the training set (printed above). These are tiny, downsampled images: the numbers say nothing about diagnostic
# performance on real radiographs.

# %% Step 4
class DeeperCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # 28 -> 14
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),       # 14 -> 7
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 2))
    def forward(self, x):
        return self.head(self.features(x))

def augment(X, rng):
    X = X.clone()
    flip = torch.tensor(rng.random(len(X)) < 0.5)
    X[flip] = X[flip].flip(-1)                                             # random horizontal flip
    shifts = rng.integers(-2, 3, size=(len(X), 2))
    return torch.stack([torch.roll(x, shifts=(int(dy), int(dx)), dims=(1, 2)) for x, (dy, dx) in zip(X, shifts)])

def train_augmented(model, X, y, epochs=8, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        Xa = augment(X, rng)                                               # a new random version each epoch
        for idx in torch.randperm(len(X)).split(64):
            opt.zero_grad()
            nn.functional.cross_entropy(model(Xa[idx]), y[idx]).backward()
            opt.step()
    return model

torch.manual_seed(0)
deeper = train(DeeperCNN(), Xp_tr, yp_tr, epochs=8)
report(deeper, Xp_te, yp_te, "Three conv layers")
torch.manual_seed(0)
deeper_aug = train_augmented(DeeperCNN(), Xp_tr, yp_tr, epochs=8)
report(deeper_aug, Xp_te, yp_te, "Three conv layers + augment")

# %% Note
# The deeper network improves AUROC (0.928 to 0.947) and, more usefully, specificity (0.538 to 0.756), at a small cost
# in sensitivity; augmentation adds a little more specificity. Compare on AUROC and on the sensitivity/specificity
# balance, not on one number. Whether flips are a
# sensible augmentation is itself a clinical question: a horizontally flipped chest X-ray puts the heart on the wrong
# side, which never happens in real images (except in dextrocardia), so small shifts are the safer augmentation here.
