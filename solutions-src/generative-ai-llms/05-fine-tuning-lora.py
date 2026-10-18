# %% Step 1
import matplotlib.pyplot as plt

ranks = [1, 2, 3, 4, 6, 8]
errors = [train_lora(r)[0] for r in ranks]
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(ranks, errors, "o-")
ax.axvline(3, linestyle=":", color="grey")
ax.set(xlabel="LoRA rank", ylabel="Training error (MSE)", title="The task needs a rank-3 change")
plt.show()
for r, e in zip(ranks, errors):
    print(f"rank {r}: training error {e:.3f}")

# %% Step 2
def fit_lora(X_tr, Y_tr, rank, steps=1500, lr=0.05, seed=1):
    r = np.random.default_rng(seed)
    A, B = r.normal(0, 0.01, (rank, d)), np.zeros((d, rank))
    for _ in range(steps):
        grad = (X_tr @ (W0 + B @ A).T - Y_tr).T @ X_tr / len(X_tr)
        B, A = B - lr * grad @ A.T, A - lr * B.T @ grad
    return A, B

rng_n = np.random.default_rng(3)
X_small, X_held = rng_n.normal(size=(150, d)), rng_n.normal(size=(1000, d))
noisy = lambda X_: X_ @ (W0 + true_change).T + rng_n.normal(0, 0.5, (len(X_), d))   # noise variance 0.25
Y_small, Y_held = noisy(X_small), noisy(X_held)

print("rank   train error   held-out error")
for rank in [1, 3, 8, 32, 64]:
    A, B = fit_lora(X_small, Y_small, rank)
    mse = lambda X_, Y_: np.mean((X_ @ (W0 + B @ A).T - Y_) ** 2)
    print(f"{rank:>4}   {mse(X_small, Y_small):>11.3f}   {mse(X_held, Y_held):>14.3f}")

# %% Note
# With only 150 noisy training examples, ranks above 3 keep lowering the training error by fitting the noise,
# while the held-out error rises above the noise floor (0.25). The rank is a capacity setting like any other:
# choose it on held-out data, and prefer the smallest rank that does the job.

# %% Step 3
A, B = fit_lora(X, Y, rank=3)
x_new = np.random.default_rng(7).normal(size=(5, d))
with_adapter = x_new @ W0.T + (x_new @ A.T) @ B.T        # base model output + adapter path
W_merged = W0 + B @ A                                      # fold the adapter into the weights
merged = x_new @ W_merged.T
print(f"Largest difference between adapter and merged outputs: {np.abs(with_adapter - merged).max():.2e}")
print("Identical up to floating-point rounding:", np.allclose(with_adapter, merged))

# %% Note
# Because the adapter is just an added matrix B @ A, it can be merged into the frozen weights after training,
# so the deployed model runs as fast as the original. Keeping it separate instead lets one base model serve
# several adapters, and makes it easy to remove an adapter that fails monitoring.

# %% Step 4
# %% Note
# Optional, with a GPU (it also runs on a CPU: about a minute on a laptop for this tiny dataset, after a
# download of about 1 GB). This sketch fine-tunes a small open instruction model
# with a LoRA adapter on the synthetic `train_pairs` from the exercise-data cell, using the Hugging Face
# `transformers` and `peft` libraries (`pip install transformers peft`). The model name is an example of a small
# open model, not a recommendation; any small causal language model works.
#
# ```python
# import torch
# from peft import LoraConfig, get_peft_model
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# tok = AutoTokenizer.from_pretrained(MODEL)
# base = AutoModelForCausalLM.from_pretrained(MODEL)
#
# def prompt(note):
#     return f"Summarise this synthetic clinic note in one or two sentences.\nNote: {note}\nSummary:"
#
# def generate(model, note, max_new_tokens=60):
#     ids = tok(prompt(note), return_tensors="pt")
#     out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
#     return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
#
# before = [generate(base, note) for note, _ in held_out_pairs]      # base model + prompting
#
# model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
#                                         target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
# model.print_trainable_parameters()
# optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
# model.train()
# for epoch in range(5):
#     for note, summary in train_pairs:
#         p_ids = tok(prompt(note), return_tensors="pt")["input_ids"]
#         full = tok(prompt(note) + " " + summary + tok.eos_token, return_tensors="pt")["input_ids"]
#         labels = full.clone()
#         labels[:, : p_ids.shape[1]] = -100           # learn only the summary, not the prompt
#         loss = model(input_ids=full, labels=labels).loss
#         loss.backward(); optimiser.step(); optimiser.zero_grad()
# model.eval()
# after = [generate(model, note) for note, _ in held_out_pairs]
#
# for (note, reference), b, a in zip(held_out_pairs, before, after):
#     print(f"NOTE      {note}\nREFERENCE {reference}\nBASE      {b}\nLORA      {a}\n")
# ```
#
# Compare the outputs by reading them against the notes, not only with an automatic score: check every
# number, medicine and plan item. With 12 training examples the adapter mostly learns the *format* and style of
# the summaries; it does not learn new medical knowledge, and it can still copy or invent details.
#
# What we saw when testing this code (October 2026, CPU; your outputs will differ with library versions):
#
# - The **base model** wrote long, generic paragraphs, called the right-toe ulcer a "left foot" ulcer, and
#   expanded GTN as "glycated hemoglobin" (it is glyceryl trinitrate).
# - The **LoRA model** produced short summaries in the reference style, but described the 63-year-old man as
#   an "older woman", and called an HbA1c of 7.6% a "blood glucose of 7.6%".
#
# Fine-tuning changed the style a lot and the factual reliability very little, which is why every output still
# needs checking against the source.

# %% Step 5
# %% Note
# Example one-page plan (a summarisation adapter for discharge letters):
#
# - **Purpose and use:** draft discharge-letter summaries for clinician editing; never sent unreviewed.
# - **Data source:** historical letters and their final edited versions from the hospital's own records, used
#   under the organisation's data-governance approval and legal basis (for example GDPR/UK GDPR, HIPAA or
#   India's DPDP Act, as applicable), with a data-protection impact assessment.
# - **De-identification:** remove direct identifiers from text before training (automated de-identification
#   plus manual checks on a sample); train inside the organisation's secure environment; do not send records
#   to outside services. Check the trained model for memorisation of rare names or phrases.
# - **Evaluation:** a held-out set from a later period and a different ward; clinician review of a random sample
#   for hallucinated and omitted facts, graded by severity; comparison with the base model plus prompting, so the
#   fine-tuning has to earn its place.
# - **Safety testing:** adversarial notes (negations, unusual units, embedded instructions), subgroup checks
#   (language, age, specialty), and a defined error threshold for stopping.
# - **Deployment approval:** the clinical safety officer and the AI/digital governance committee, with a clinical
#   safety case; regulatory assessment of whether the tool is a medical device in the intended use.
# - **After deployment:** monitor edit rates and reported errors monthly; retrain or withdraw on agreed triggers.
