# %% Step 1
for k in range(2, 7):
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    print(f"k = {k}: silhouette {silhouette_score(X, lab):.2f}")
labels3 = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
print(f"k = 3 agreement with the true groups (ARI) {adjusted_rand_score(df.true_group, labels3):.2f}")

# %% Step 2
def make_profiles(spread_factor, seed=17):
    rng = np.random.default_rng(seed)
    frames = [pd.DataFrame({k: rng.normal(v, spread[k] * spread_factor, 600) for k, v in c.items()})
              .assign(true_group=name) for name, c in centres.items()]
    return pd.concat(frames, ignore_index=True)

for factor in [1.0, 1.5, 2.0, 3.0]:
    d = make_profiles(factor)
    Xd = StandardScaler().fit_transform(d[variables])
    lab = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Xd)
    print(f"spread × {factor}: silhouette {silhouette_score(Xd, lab):.2f}, "
          f"agreement with truth (ARI) {adjusted_rand_score(d.true_group, lab):.2f}")

# %% Note
# As the groups overlap, both numbers fall. Real patient data usually look like the more overlapping
# settings: a continuum with denser regions, not separate islands. Low silhouette values are a warning that
# cluster labels will put many patients in a group they barely belong to.

# %% Step 3
rng = np.random.default_rng(0)
reference = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
scores = []
for b in range(50):
    idx = rng.choice(len(X), len(X), replace=True)
    boot = KMeans(n_clusters=3, n_init=10, random_state=b).fit(X[idx])
    scores.append(adjusted_rand_score(reference.predict(X), boot.predict(X)))
print(f"Bootstrap stability, k = 3: mean ARI {np.mean(scores):.2f} (range {min(scores):.2f}-{max(scores):.2f})")

# The same check for a k that the data do not support
ref5 = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X)
scores5 = [adjusted_rand_score(ref5.predict(X), KMeans(n_clusters=5, n_init=10, random_state=b)
                               .fit(X[rng.choice(len(X), len(X), replace=True)]).predict(X)) for b in range(50)]
print(f"Bootstrap stability, k = 5: mean ARI {np.mean(scores5):.2f}")

# %% Note
# With three real, well-separated groups, k = 3 is found again on almost every resample (ARI close to 1).
# Forcing five clusters splits real groups in arbitrary places that change from resample to resample, so
# stability drops. Stability is a useful check, but not proof: the pure-noise example in the lesson shows
# that unstable clusters are a warning sign, while stable ones still need clinical meaning.

# %% Step 4
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

methods = {
    "k-means": labels3,
    "Gaussian mixture": GaussianMixture(n_components=3, random_state=0).fit_predict(X),
    "Hierarchical (Ward)": AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X),
}
names = list(methods)
print("Agreement between methods (ARI):")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print(f"  {names[i]} vs {names[j]}: {adjusted_rand_score(methods[names[i]], methods[names[j]]):.2f}")
for name, lab in methods.items():
    print(f"{name:<20} vs truth: {adjusted_rand_score(df.true_group, lab):.2f}")

# %% Note
# On these well-separated groups the three methods agree closely with each other and with the truth. A good
# further experiment: repeat the comparison on the overlapping data from step 2 (spread × 2 or × 3). If the
# methods then disagree, the "clusters" reflect each method's assumptions more than the patients.

# %% Step 5
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(5)
logit = -4 + 0.6 * (df.hba1c - 8) + 0.5 * (df.homa_ir - 3)
df["complication"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
df["cluster"] = labels3.astype(str)
print(f"Complication rate {df.complication.mean():.1%}")

train, test = train_test_split(df, test_size=0.3, random_state=0, stratify=df.complication)
X_cl_train, X_cl_test = pd.get_dummies(train.cluster), pd.get_dummies(test.cluster)
cluster_model = LogisticRegression().fit(X_cl_train, train.complication)
continuous_model = LogisticRegression(max_iter=1000).fit(train[variables], train.complication)
print(f"Cluster labels only:  AUROC {roc_auc_score(test.complication, cluster_model.predict_proba(X_cl_test)[:, 1]):.3f}")
print(f"Continuous variables: AUROC {roc_auc_score(test.complication, continuous_model.predict_proba(test[variables])[:, 1]):.3f}")

# %% Note
# Cluster labels throw away the variation inside each cluster: two patients in the same cluster with HbA1c
# 8% and 12% get the same prediction. For predicting an individual's outcome, the continuous measurements
# usually do better; clusters can still be useful for describing a population or designing trials.
