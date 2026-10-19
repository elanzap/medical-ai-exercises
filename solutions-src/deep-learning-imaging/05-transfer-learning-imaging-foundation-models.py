# %% Step 1
import matplotlib.pyplot as plt

def freeze_and_retrain_head(base, X, y, epochs=30):
    m = copy.deepcopy(base)
    for p in m.features.parameters():
        p.requires_grad = False
    m.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, 2))
    torch.manual_seed(0)
    opt = torch.optim.Adam(m.head.parameters(), lr=1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(X)).split(64):
            opt.zero_grad(); nn.functional.cross_entropy(m(X[idx]), y[idx]).backward(); opt.step()
    return m

as_is = accuracy(pretrained, X_target_test, y_target_test)
rows = []
for n_b in [25, 100, 500]:
    Xb, yb = new_scanner(n_b, seed=20 + n_b)
    torch.manual_seed(0)
    rows.append({"site-B images": n_b, "applied as is": as_is,
                 "from scratch": accuracy(train(SmallCNN(), Xb, yb, epochs=30), X_target_test, y_target_test),
                 "frozen features": accuracy(freeze_and_retrain_head(pretrained, Xb, yb), X_target_test, y_target_test),
                 "fine-tuned": accuracy(train(copy.deepcopy(pretrained), Xb, yb, epochs=30), X_target_test, y_target_test)})
import pandas as pd
curve = pd.DataFrame(rows).set_index("site-B images")
print(curve.round(3).to_string())

fig, ax = plt.subplots(figsize=(5.5, 3.5))
for col in curve.columns:
    ax.plot(curve.index, curve[col], "o-", label=col)
ax.set(xscale="log", xlabel="Labelled site-B images", ylabel="Site-B accuracy", xticks=curve.index)
ax.set_xticklabels(curve.index); ax.legend(fontsize=8)
plt.show()

# %% Note
# With few labelled images, starting from the pretrained model helps most: at 25 site-B images, training from scratch
# is no better than chance (0.511) while frozen features reach 0.672. As labelled site-B data grow, the gap narrows
# (fine-tuned minus scratch: 0.18 at 100 images, 0.10 at 500) but has not closed. That is the general pattern behind foundation models: they matter most where labels are
# scarce, which in medicine is most of the time.

# %% Step 2
def standardise(X):
    """Per-site normalisation: subtract that site's mean and divide by its standard deviation."""
    return (X - X.mean()) / X.std()

pretrained_std = train(SmallCNN(), standardise(X_source), y_source)
print(f"Pretrained on raw site-A images, site B as is:          {as_is:.3f}")
print(f"Pretrained on standardised site A, standardised site B: "
      f"{accuracy(pretrained_std, standardise(X_target_test), y_target_test):.3f}")

# %% Note
# The site-B shift here is purely a contrast change and an intensity offset, so standardising each site's images
# removes it, and the pretrained model works at site B with no fine-tuning at all. Real scanner differences (noise
# texture, resolution, artefacts, different patient mix) are rarely this simple, but intensity normalisation is a
# cheap first step that should always be tried, and checked, before fine-tuning.

# %% Step 3
# Optional: an ImageNet-pretrained ResNet-18 (torchvision) on RetinaMNIST, compared with the same network from scratch.
# Downloads about 45 MB of weights and 3 MB of images; a few minutes on a CPU.
from medmnist import RetinaMNIST
from torchvision.models import ResNet18_Weights, resnet18

def load_retina(split):
    d = RetinaMNIST(split=split, download=True, root="data", size=28)
    X = torch.tensor(d.imgs, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    X = nn.functional.interpolate(X, size=64, mode="bilinear", align_corners=False)   # small images, upsampled
    mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (X - mean) / std, torch.tensor(d.labels.ravel(), dtype=torch.long)

Xr_tr, yr_tr = load_retina("train")
Xr_te, yr_te = load_retina("test")

def resnet(pretrained_weights):
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained_weights else None)
    m.fc = nn.Linear(m.fc.in_features, 5)                    # five retinopathy grades (0-4)
    return m

def fit_eval(m, epochs=6, lr=1e-3):
    torch.manual_seed(0)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    m.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(Xr_tr)).split(64):
            opt.zero_grad(); nn.functional.cross_entropy(m(Xr_tr[idx]), yr_tr[idx]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(Xr_te).argmax(1)
    exact = (pred == yr_te).float().mean().item()
    within1 = ((pred - yr_te).abs() <= 1).float().mean().item()
    return exact, within1

for label, weights in [("ResNet-18 from scratch", False), ("ResNet-18, ImageNet-pretrained", True)]:
    exact, within1 = fit_eval(resnet(weights))
    print(f"{label:<32} exact grade {exact:.3f}   within one grade {within1:.3f}")

# %% Note
# RetinaMNIST has only 1,080 training images, graded 0-4 (MedMNIST, CC BY 4.0), so it is hard at 28x28 and accuracy
# is modest for both networks. Grading is ordinal, which is why "within one grade" is reported too. In our run the
# ImageNet-pretrained network was slightly better on the exact grade (0.527 against 0.498) and no better within one
# grade (0.765 against 0.770): a small, uncertain gain on 400 test images. Results vary with seeds and epochs on a
# dataset this small, so compare on your own run rather than assuming pretraining helps.

# %% Step 4
# %% Note
# Example validation plan before using an imaging foundation model in a diabetic retinopathy screening programme:
#
# 1. **Intended use** written down: the grading task (referable retinopathy and maculopathy, or full grading), the
#    population, the cameras, and whether the model replaces, assists or quality-checks human graders.
# 2. **Regulatory status** for that intended use, and the manufacturer's evidence: what data the foundation model
#    and the final classifier were trained on, and whether any of our own images were included.
# 3. **Local retrospective validation** on a large, consecutive sample from our programme (several thousand images,
#    with enough referable cases), graded by certified graders with arbitration as the reference standard:
#    sensitivity for referable disease (the priority), specificity, ungradable rate, with confidence intervals.
# 4. **Subgroups**: each camera model and site, mydriatic versus non-mydriatic images, age, sex, ethnicity, and image
#    quality; acceptance criteria for each, set in advance.
# 5. **Comparison with current practice**: the programme's existing grader performance on the same images.
# 6. **Prospective silent run**: the model grades alongside the usual pathway for a few months, without affecting care.
# 7. **Failure handling**: ungradable and borderline images always go to a human grader; no image is discarded unseen.
# 8. **Monitoring after go-live**: referral rates, ungradable rates and a graded random sample each month; re-validation
#    after any model update or new camera; an owner and a switch-off rule.
