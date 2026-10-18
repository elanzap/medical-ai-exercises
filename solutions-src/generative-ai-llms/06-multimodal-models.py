# %% Step 1
print(f"Zero-shot accuracy (no shortcut): {zero_shot_accuracy(imgs, labels):.1%}")
print("artefact in pneumonia   same hospital   new hospital")
for p_tag in [0.1, 0.5, 0.7, 0.9, 0.99]:
    d_imgs, d_lab = make_images(3000, [0.1, p_tag, 0.1])
    clf = LogisticRegression(max_iter=2000).fit(d_imgs, d_lab)
    s_imgs, s_lab = make_images(3000, [0.1, p_tag, 0.1])
    n_imgs, n_lab = make_images(3000, [0.5, 0.5, 0.5])
    print(f"{p_tag:>20.0%}   {clf.score(s_imgs, s_lab):>13.1%}   {clf.score(n_imgs, n_lab):>12.1%}")

# %% Note
# The more tightly the scanner artefact is tied to pneumonia at the development hospital, the better the
# classifier looks there and the worse it does at a hospital where the scanner is unrelated to diagnosis. The
# same-hospital test cannot reveal the problem, because the shortcut works there too.

# %% Step 2
def make_images_tagged(n, p_tag_by_class):
    """Same as make_images, but also returns which images carry the artefact (e.g. from scanner metadata)."""
    labs = rng.integers(0, 3, n)
    tagged = rng.random(n) < np.array(p_tag_by_class)[labs]
    raw = concepts[labs] + rng.normal(0, 3.0, (n, dim)) + 3.0 * tagged[:, None] * hospital_tag
    return normalise(raw), labs, tagged

dev_x, dev_y, dev_tag = make_images_tagged(3000, [0.1, 0.9, 0.1])
new_x, new_y, _ = make_images_tagged(3000, [0.5, 0.5, 0.5])

# Estimate the artefact direction from the data: mean difference between tagged and untagged images
# within each class, so the disease signal is not removed along with it.
direction = np.mean([dev_x[(dev_y == c) & dev_tag].mean(0) - dev_x[(dev_y == c) & ~dev_tag].mean(0)
                     for c in range(3)], axis=0)
u = direction / np.linalg.norm(direction)
project_out = lambda x: x - np.outer(x @ u, u)
print(f"Cosine similarity of estimated and true artefact direction: {u @ hospital_tag / np.linalg.norm(hospital_tag):.2f}")

before = LogisticRegression(max_iter=2000).fit(dev_x, dev_y)
after = LogisticRegression(max_iter=2000).fit(project_out(dev_x), dev_y)
print(f"New hospital accuracy, original embeddings:      {before.score(new_x, new_y):.1%}")
print(f"New hospital accuracy, artefact projected out:   {after.score(project_out(new_x), new_y):.1%}")

# %% Note
# Removing the artefact direction takes the shortcut away, so the classifier has to use the disease signal,
# and it transfers better. In real images the artefact is rarely a single clean direction (it may be text
# markers, positioning, image processing), so this is a diagnostic idea more than a fix: the reliable
# remedies are training data where the artefact is not tied to the diagnosis, and testing at new sites.

# %% Step 3
def zero_shot_with(text_emb, x, y):
    return ((x @ text_emb.T).argmax(axis=1) == y).mean()

test_x, test_y = make_images(3000, [0.1, 0.1, 0.1])
r = np.random.default_rng(30)
singles = [zero_shot_with(normalise(concepts + r.normal(0, 0.2, (3, dim))), test_x, test_y) for _ in range(20)]
for n_prompts in [1, 3, 5, 10]:
    accs = []
    for _ in range(20):
        prompts = concepts[None] + r.normal(0, 0.2, (n_prompts, 3, dim))   # several descriptions per class
        accs.append(zero_shot_with(normalise(normalise(prompts).mean(axis=0)), test_x, test_y))
    print(f"{n_prompts:>2} prompt(s) per class: mean zero-shot accuracy {np.mean(accs):.1%} (sd {np.std(accs):.1%})")

# %% Note
# Each text prompt is a noisy description of its class. Averaging several cancels part of that noise, so
# zero-shot accuracy rises a little and, more noticeably, varies less with the exact wording chosen. With
# real models, prompt ensembling uses paraphrases such as "chest X-ray with pneumonia", "radiograph showing
# consolidation", "lobar pneumonia on a chest film".

# %% Step 4
# %% Note
# Optional extension (outline, not run here). With an openly licensed chest X-ray dataset and an open
# vision-language model:
#
# 1. **Check the licence and terms** of both the dataset and the model, and use only de-identified images.
# 2. **Pick two sites or acquisition types** recorded in the metadata (for example two hospitals, or frontal
#    images taken with portable and fixed scanners), and a small set of labels present in both.
# 3. **Write two or three text prompts per label**, embed the images and prompts with the model, and classify
#    each image by the most similar averaged prompt (as in step 3).
# 4. **Report accuracy (or AUROC per label) separately for each site/acquisition type**, with confidence
#    intervals, and compare the label mix at each site: a difference in accuracy can come from case mix as well
#    as from image differences.
#
# A large gap between sites is the real-world version of the shortcut in step 1, and a reason to validate any
# imaging model at each site where it will be used.
