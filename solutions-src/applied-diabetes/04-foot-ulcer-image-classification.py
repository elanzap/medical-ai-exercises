# %% Step 1
import matplotlib.pyplot as plt

shares, light, dark_ = [0.05, 0.15, 0.30, 0.50], [], []
for share in shares:
    X, y, _ = wound_photos(4000, seed=10 + int(share * 100), dark_share=share)
    l, d = accuracy_by_tone(train(X, y), X_te, y_te, dark_te)
    light.append(l); dark_.append(d)
    print(f"{share:>4.0%} darker skin in training: lighter {l:.1%}, darker {d:.1%}")
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot([100 * s for s in shares], light, "o-", label="lighter skin")
ax.plot([100 * s for s in shares], dark_, "o-", label="darker skin")
ax.set(xlabel="Darker-skin photos in training (%)", ylabel="Test accuracy", ylim=(0.5, 1.01)); ax.legend()
plt.show()

# %% Note
# With this training sample, even 5% darker-skin photos gave 92% accuracy on darker skin, against 71.6% in the lesson's
# run with a different random sample. Performance on an under-represented group is unstable as well as lower: it depends
# on which few examples happened to be included. More examples make it both higher and more predictable.

# %% Step 2
def wound_photos_v2(n, seed, dark_share, dark_contrast=0.5):
    """As wound_photos, but ulcers on darker skin have lower contrast (scaled by dark_contrast)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:28, :28]
    dark = rng.random(n) < dark_share
    tone = np.where(dark, rng.uniform(-1.6, -0.8, n), rng.uniform(0.8, 1.6, n))
    images = tone[:, None, None] + rng.normal(0, 0.6, (n, 28, 28))
    labels = rng.integers(0, 2, n)
    for i in np.where(labels == 1)[0]:
        cy, cx, r = rng.integers(7, 21), rng.integers(7, 21), rng.uniform(3, 5)
        patch = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        depth = 1.2 * (dark_contrast if dark[i] else 1.0)
        images[i][patch] -= depth + rng.normal(0, 0.4 * (dark_contrast if dark[i] else 1.0), patch.sum())
    return torch.tensor(images, dtype=torch.float32).unsqueeze(1), torch.tensor(labels), dark

X2_tr, y2_tr, _ = wound_photos_v2(4000, seed=21, dark_share=0.05)
X2_te, y2_te, dark2_te = wound_photos_v2(2000, seed=22, dark_share=0.5)
for label, Xa, Xb in [("raw images", X2_tr, X2_te), ("normalised", normalise(X2_tr), normalise(X2_te))]:
    l, d = accuracy_by_tone(train(Xa, y2_tr), Xb, y2_te, dark2_te)
    print(f"Lower ulcer contrast on darker skin, {label:<11}: lighter {l:.1%}, darker {d:.1%}")
X2_bal, y2_bal, _ = wound_photos_v2(4000, seed=23, dark_share=0.5)
l, d = accuracy_by_tone(train(normalise(X2_bal), y2_bal), normalise(X2_te), y2_te, dark2_te)
print(f"Lower contrast, normalised AND 50% darker skin in training: lighter {l:.1%}, darker {d:.1%}")

# %% Note
# When the only difference was overall brightness, normalisation removed it completely. When ulcers really are harder
# to see on darker skin (lower contrast), normalisation cannot restore information that is not in the image as strongly;
# a gap remains, and diverse training data help more. In real photography, lighting, camera exposure and the way
# erythema shows on different skin tones all contribute, so performance must be measured, not assumed.

# %% Step 3
def wilson(k, n, z=1.96):
    p = k / n
    c, h = (p + z**2 / (2 * n)) / (1 + z**2 / n), z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return c - h, c + h

model2 = train(normalise(X2_tr), y2_tr)
with torch.no_grad():
    pred = model2(normalise(X2_te)).argmax(1).numpy()
truth = y2_te.numpy()
for label, m in [("lighter skin", ~dark2_te), ("darker skin", dark2_te)]:
    tp = np.sum((pred == 1) & (truth == 1) & m); fn = np.sum((pred == 0) & (truth == 1) & m)
    tn = np.sum((pred == 0) & (truth == 0) & m); fp = np.sum((pred == 1) & (truth == 0) & m)
    (sl, sh), (pl, ph) = wilson(tp, tp + fn), wilson(tn, tn + fp)
    print(f"{label:<13} sensitivity {tp / (tp + fn):.1%} ({sl:.1%}-{sh:.1%}), specificity {tn / (tn + fp):.1%} ({pl:.1%}-{ph:.1%})")

# %% Note
# A minimum standard should be set in advance for every group, not only overall: for example, sensitivity for ulcers of
# at least 90% with the lower confidence limit above 85% in each skin-tone group, and no group more than 5 points
# below the best. A model that fails in one group fails the standard, however good the average is; the fix is better
# data for that group, not a lower bar.

# %% Step 4
def healing_check(areas_cm2, weeks):
    """Flag growth between consecutive visits, and slow healing at 4 weeks (under 50% area reduction)."""
    messages = []
    for (w0, a0), (w1, a1) in zip(zip(weeks, areas_cm2), zip(weeks[1:], areas_cm2[1:])):
        if a1 > a0 * 1.10:                                   # grown by more than 10%
            messages.append(f"Week {w1}: ulcer area increased from {a0:.1f} to {a1:.1f} cm2 (+{a1 / a0 - 1:.0%}) "
                            f"since week {w0}. Review today for infection, ischaemia or offloading problems.")
    if 4 in weeks and areas_cm2[weeks.index(4)] > 0.5 * areas_cm2[0]:
        change = 1 - areas_cm2[weeks.index(4)] / areas_cm2[0]
        trend = f"reduced by only {change:.0%}" if change >= 0 else f"{-change:.0%} LARGER than at week 0"
        messages.append(f"Week 4: area {trend} (target: at least 50% smaller). Consider escalating care.")
    return messages or ["Healing on track."]

cases = {"patient A": [4.0, 3.1, 2.4, 1.7, 1.1], "patient B": [3.5, 3.4, 3.3, 3.0, 2.9],
         "patient C": [6.2, 5.6, 4.0, 3.5, 3.0], "patient D": [2.0, 1.8, 2.4, 2.6, 2.5]}
for name, a in cases.items():
    print(name)
    for msg in healing_check(a, weeks=[0, 1, 2, 3, 4]):
        print("   ", msg)

# %% Note
# The message a clinician sees should state the measurement, the change and the time, and suggest what to check, while
# leaving the decision to them. Because the area comes from automatic segmentation, the message should also link to the
# photographs, so a clinician can see whether the "growth" is real or a change in camera angle or lighting.
