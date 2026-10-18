# %% Step 1
# The worked example above already generated `df` (1,000 synthetic patients) and defined k_anonymity().
k, unique = k_anonymity(df, quasi_identifiers)
print(f"Raw data: k = {k}, unique people = {unique} of {len(df)}")

# %% Step 2
def band(ages, width):
    lower = ages // width * width
    return lower.astype(str) + "-" + (lower + width - 1).astype(str)

schemes = {
    "5-year age bands": df.assign(age=band(df.age, 5)),
    "10-year age bands": df.assign(age=band(df.age, 10)),
    "3-digit postcodes": df.assign(postcode=df.postcode.str[:3]),
    "10-year bands + 3-digit postcodes": df.assign(age=band(df.age, 10), postcode=df.postcode.str[:3]),
}
results = []
for name, data in schemes.items():
    k, unique = k_anonymity(data, quasi_identifiers)
    groups = data.groupby(quasi_identifiers).ngroups
    results.append({"scheme": name, "k": k, "unique people": unique, "groups": groups})
print(pd.DataFrame(results).to_string(index=False))

# %% Note
# All the synthetic postcodes share their first three digits, so "3-digit postcodes" removes the
# postcode as an identifier entirely. With real postcodes the trade-off is less extreme, but the pattern
# is the same: each generalisation raises k and loses detail. Age bands on their own are not enough here,
# because 40 postcodes × 2 sexes × several bands still leaves many groups of one or two people.

# %% Step 3
preferred = schemes["10-year bands + 3-digit postcodes"]
diversity = preferred.groupby(quasi_identifiers).diagnosis.nunique()
print(f"Groups: {len(diversity)}, smallest number of distinct diagnoses in a group (l) = {diversity.min()}")
print("Groups with only one diagnosis:", int((diversity == 1).sum()))

# The finer 5-year scheme, for comparison:
fine = schemes["5-year age bands"].groupby(quasi_identifiers).diagnosis.nunique()
print(f"5-year bands: l = {fine.min()}, groups with a single diagnosis = {int((fine == 1).sum())} of {len(fine)}")

# %% Note
# A group in which everyone has the same diagnosis leaks that diagnosis to anyone who knows a person is in
# the group, even when k is large. That is why l-diversity is checked alongside k-anonymity.

# %% Step 4
rng = np.random.default_rng(5)
n_patients = 50
visit_table = pd.DataFrame({
    "patient_id": np.repeat([f"p{i:02d}" for i in range(n_patients)], 4),
    "visit_date": pd.Timestamp("2025-01-01")
                  + pd.to_timedelta(rng.integers(0, 400, size=n_patients * 4), unit="D"),
}).sort_values(["patient_id", "visit_date"], ignore_index=True)

shift = {pid: pd.Timedelta(days=int(rng.integers(-180, 181))) for pid in visit_table.patient_id.unique()}
visit_table["shifted_date"] = visit_table.visit_date + visit_table.patient_id.map(shift)

gap_original = visit_table.groupby("patient_id").visit_date.diff()
gap_shifted = visit_table.groupby("patient_id").shifted_date.diff()
print("Intervals between visits preserved for every patient:", gap_original.equals(gap_shifted))
print(visit_table.head(4))

# %% Note
# What is *not* preserved: calendar effects (season, weekday, public holidays) and the timing of one
# patient's visits relative to another's. If an analysis depends on those, date shifting is the wrong tool.

# %% Step 5
# %% Note
# Example note. De-identification lowers the risk; it does not remove it. Before sharing with an external
# AI developer I would also want:
#
# - **A data-sharing agreement** that names the purpose, forbids re-identification and linkage with other
#   data, limits onward sharing, and sets deletion at the end of the project.
# - **The legal basis checked** under the law that applies (for example HIPAA, the UK GDPR / EU GDPR, or
#   India's DPDP Act), plus ethics or data-governance approval.
# - **Data minimisation:** only the fields the model needs, at the coarsest level of detail that works.
# - **A controlled environment** (a trusted research environment or secure enclave) so the data are
#   analysed in place rather than copied, with access logging and output checks.
# - **Free-text and images reviewed separately**, because names and dates hide in notes and image headers.
# - **A re-identification risk assessment** repeated whenever new fields are added.
