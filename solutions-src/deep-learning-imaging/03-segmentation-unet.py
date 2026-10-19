# %% Step 1
import matplotlib.pyplot as plt

with torch.no_grad():
    logits = unet(X_test)
    d = dice((logits > 0).float(), M_test)
worst = torch.argsort(d)[:5]
fig, axes = plt.subplots(3, 5, figsize=(12, 7))
for col, i in enumerate(worst):
    for row, (img, title) in enumerate([(X_test[i, 0], "scan"), (M_test[i, 0], "true mask"),
                                        ((logits[i, 0] > 0).float(), f"predicted, Dice {d[i]:.2f}")]):
        axes[row, col].imshow(img, cmap="gray"); axes[row, col].set_title(title, fontsize=8); axes[row, col].axis("off")
plt.tight_layout(); plt.show()
print("Worst five Dice scores:", [round(float(d[i]), 3) for i in worst])

# %% Note
# The worst five include a complete miss (Dice 0.0) and several partial ones. Look at the figure to see what they have
# in common; small or faint lesions, where a few wrong pixels change Dice a lot, are a common cause. Looking at the
# failures is how you learn what a mean of 0.89 hides.

# %% Step 2
def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)).mean()

def train_unet(loss_name, X=X_train, M=M_train, model_cls=TinyUNet, epochs=8, report_epochs=(1, 4, 8)):
    torch.manual_seed(0)
    net = model_cls()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    history = {}
    for epoch in range(1, epochs + 1):
        for idx in torch.randperm(len(X)).split(32):
            opt.zero_grad()
            out = net(X[idx])
            bce = nn.functional.binary_cross_entropy_with_logits(out, M[idx])
            loss = bce if loss_name == "BCE" else bce + dice_loss(out, M[idx])
            loss.backward(); opt.step()
        if epoch in report_epochs:
            with torch.no_grad():
                history[epoch] = dice((net(X_test) > 0).float(), M_test).mean().item()
    return net, history

for name in ["BCE", "BCE + Dice"]:
    _, hist = train_unet(name)
    print(f"{name:<11}", "  ".join(f"epoch {e}: {v:.3f}" for e, v in hist.items()))

# %% Note
# Lesion pixels are a small fraction of each scan, so plain cross-entropy is dominated by the background and starts by
# predicting "no lesion" everywhere (Dice 0 after the first epoch). Adding a Dice term rewards overlap directly. Here
# both losses are still at 0 after one epoch; the combined loss is slightly ahead at epoch 4 (0.876 against 0.864) and
# at the end (0.895 against 0.891). The benefit of a Dice term is usually larger when lesions are rarer than this.

# %% Step 3
def make_small_lesion_scans(n, seed, size=32):
    """As make_scans, but each scan has one normal lesion and one very small one (radius 1-1.6 pixels)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    images, masks, small = rng.normal(0, 1, (n, size, size)), np.zeros((n, size, size)), np.zeros((n, size, size))
    for i in range(n):
        for radius_range, store in [((2, 5), None), ((1.0, 1.6), small)]:
            cy, cx = rng.integers(6, size - 6, 2)
            ry, rx = rng.uniform(*radius_range, 2)
            blob = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
            masks[i] = np.maximum(masks[i], blob)
            if store is not None:
                store[i] = blob
        images[i] += 1.5 * masks[i]
    to_t = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(1)
    return to_t(images), to_t(masks), to_t(small)

Xs_test, Ms_test, small_test = make_small_lesion_scans(300, seed=7)
with torch.no_grad():
    pred = (unet(Xs_test) > 0).float()
mean_dice = dice(pred, Ms_test).mean().item()
detected = ((pred * small_test).sum(dim=(1, 2, 3)) > 0).float().mean().item()   # any overlap counts as detected
print(f"Mean Dice with small lesions present: {mean_dice:.3f}")
print(f"Small lesions detected (any overlap): {detected:.1%}")

# %% Note
# Mean Dice falls only modestly (from 0.891 to 0.826), because the small lesions contribute few pixels to each scan's
# overlap, while the per-lesion detection rate shows that most of them (about 69%) are missed. In screening for small lesions (for example
# microaneurysms in diabetic retinopathy), detection rate is the clinically important number, so report both.

# %% Step 4
import time

class TwoLevelUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = block(1, 8), block(8, 16), block(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.up2, self.dec2 = nn.ConvTranspose2d(32, 16, 2, stride=2), block(32, 16)
        self.up1, self.dec1 = nn.ConvTranspose2d(16, 8, 2, stride=2), block(16, 8)
        self.out = nn.Conv2d(8, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

for name, cls in [("one downsampling level", TinyUNet), ("two downsampling levels", TwoLevelUNet)]:
    start = time.perf_counter()
    net, hist = train_unet("BCE + Dice", model_cls=cls, report_epochs=(8,))
    secs = time.perf_counter() - start
    params = sum(p.numel() for p in net.parameters())
    print(f"{name:<24} final Dice {hist[8]:.3f}  parameters {params:>6,}  training time {secs:.0f} s")

# %% Note
# The second level sees a wider context (each output pixel depends on a larger part of the image) at the cost of
# more parameters and time. On 32x32 scans with small lesions there is little extra context to use, so gains are
# modest; on large images with big structures, depth matters more. Timing depends on your hardware.

# %% Step 5
final_dice = d
print(f"n={len(final_dice)}, mean {final_dice.mean():.3f}, median {final_dice.median():.3f}, "
      f"IQR {torch.quantile(final_dice, 0.25):.3f}-{torch.quantile(final_dice, 0.75):.3f}, "
      f"min {final_dice.min():.3f}, scans below 0.7: {(final_dice < 0.7).sum().item()}")

# %% Note
# Example results paragraph (fill in the numbers printed above):
#
# "On 300 held-out synthetic scans, the U-Net achieved a mean Dice coefficient of 0.89 (median 0.91, interquartile
# range 0.87-0.93), compared with 0.27 for a brightness-threshold baseline. Performance was not uniform: four scans
# scored below 0.7, including one complete miss (examples in the figure). When a second, very small lesion was added
# to each scan, only 31% of those small lesions were detected, although mean Dice fell only to 0.83. Results are on synthetic data from the same generator as the training set, so they
# indicate whether the method works, not how it would perform on real images."
