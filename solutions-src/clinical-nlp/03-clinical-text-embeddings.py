# %% Step 1
p_tfidf = clf.predict_proba(tfidf.transform(test_notes))[:, 1]
p_emb = clf_e.predict_proba(E_test)[:, 1]
print(f"TF-IDF AUROC {roc_auc_score(y_test, p_tfidf):.3f}; embeddings AUROC {roc_auc_score(y_test, p_emb):.3f}")

def probe(model_predict, label):
    print(label)
    for text, p in zip(probes, model_predict(probes)):
        print(f"   P(foot problem) {p:.2f}  {text}")

probe(lambda t: clf_e.predict_proba(encoder.encode(t, normalize_embeddings=True))[:, 1], "Embedding model:")

# %% Step 2
negated_phrases = ["no tingling in the feet", "denies numbness in the toes", "no foot ulcers",
                   "feet examined, sensation normal", "no burning pain in the feet"]
rng_neg = np.random.default_rng(1)
neg_notes = [f"{rng_neg.choice(openers)} Patient reports {rng_neg.choice(negated_phrases)}. HbA1c {rng_neg.uniform(6, 10):.1f}%."
             for _ in range(80)]
train_notes_2 = train_notes + neg_notes
y_train_2 = np.concatenate([y_train, np.zeros(len(neg_notes), dtype=int)])
clf_e2 = LogisticRegression(max_iter=1000).fit(encoder.encode(train_notes_2, normalize_embeddings=True), y_train_2)

probe(lambda t: clf_e2.predict_proba(encoder.encode(t, normalize_embeddings=True))[:, 1], "Retrained with negated notes:")
print(f"Paraphrase test AUROC after retraining {roc_auc_score(y_test, clf_e2.predict_proba(E_test)[:, 1]):.3f}")

# %% Note
# Adding negated examples labelled "no problem" lowers the negated probes (0.90 to 0.51, and 0.76 to 0.26), but at a
# cost: the genuine "reports tingling" probe falls to 0.60 and the paraphrase test AUROC drops from 1.000 to 0.886.
# The embeddings place "no tingling in the feet" close to "tingling in the feet", so the classifier can only lower
# confidence for anything foot-related. Real negation comes in many forms, and teaching it through examples alone needs
# far more data than this.

# %% Step 3
import re

NEGATION_CUES = r"\b(no|denies|denied|without|negative for|absent)\b"

def drop_negated_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(s for s in sentences if not re.search(NEGATION_CUES, s.lower()))

filtered_probes = [drop_negated_sentences(p) for p in probes]
for original, filtered, p in zip(probes, filtered_probes,
                                 clf_e.predict_proba(encoder.encode(filtered_probes, normalize_embeddings=True))[:, 1]):
    print(f"P(foot problem) {p:.2f}  {filtered!r}  (from {original!r})")
filtered_test = [drop_negated_sentences(t) for t in test_notes]
print(f"Paraphrase test AUROC with negated sentences removed "
      f"{roc_auc_score(y_test, clf_e.predict_proba(encoder.encode(filtered_test, normalize_embeddings=True))[:, 1]):.3f}")

# %% Note
# Removing negated sentences before embedding fixes the probes without retraining: only "Annual review. HbA1c 7.2%."
# remains, which carries no foot information. It is a blunt tool: a sentence such as "no ulcers but numbness in both
# feet" would be dropped entirely, losing the positive finding. Combining a negation step with negated training
# examples is more robust than either alone.

# %% Step 4
# Optional: a clinical encoder (Bio_ClinicalBERT, about 430 MB) with mean pooling, compared on the paraphrase test.
import torch
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").eval()

def mean_pooled(texts, batch=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=64, return_tensors="pt")
            hidden = bert(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            out.append(((hidden * mask).sum(1) / mask.sum(1)).numpy())   # average over real tokens only
    return np.vstack(out)

B_train, B_test = mean_pooled(train_notes), mean_pooled(test_notes)
clf_b = LogisticRegression(max_iter=2000).fit(B_train, y_train)
print(f"Bio_ClinicalBERT mean-pooled + logistic regression: paraphrase test AUROC "
      f"{roc_auc_score(y_test, clf_b.predict_proba(B_test)[:, 1]):.3f}")
print(f"all-MiniLM-L6-v2 sentence embeddings (for comparison):     "
      f"{roc_auc_score(y_test, clf_e.predict_proba(E_test)[:, 1]):.3f}")

# %% Note
# Bio_ClinicalBERT was pretrained on clinical notes (MIMIC-III), but as a masked-language model, not to produce
# sentence embeddings; simply averaging its token vectors often gives weaker sentence-level similarity than a model
# trained for that purpose. Domain pretraining helps most when the encoder is fine-tuned on the task, or trained
# specifically for sentence similarity, so compare rather than assume.
