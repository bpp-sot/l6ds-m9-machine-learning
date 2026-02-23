# Reference: Data Quality Checklist

Prior to entering ANY Modeling or Feature Selection pipeline frameworks within your L6 Apprenticeship presentation, guarantee you have audited against the following Data Quality dimensions. Failure to map these dimensions implies gross negligence of the core mathematical principles inherent in Data Science practices. 

## 1. Completeness
Are records entirely intact, or are fragments missing?

- [ ] Evaluated columns carrying explicit `np.nan` artifacts.
- [ ] Confirmed zero 'Phantom representations' exist (i.e. 'N/A', '-99', '?', ' ').
- [ ] Justified Missing Value treatment protocols explicitly (`SimpleImputer` vs Deletion).

## 2. Uniqueness
Do duplicated identifiers bloat your sampling space?

- [ ] Assessed exactly duplicated rows (`df.duplicated()`).
- [ ] Evaluated multi-key identity collisions resulting from erratic joins (`pd.merge()` without index evaluation).
- [ ] Validated any required Master/Detail mappings.

## 3. Consistency
Are data entries stored using synchronized standards? 

- [ ] Confirmed text is purely lower/uppercase (no case discrepancies).
- [ ] Validated trailing or leading whitespace artifacts strings stripped via `.str.strip()`.
- [ ] Converted erratic timestamp markers to standard ISO-8601 formatting or standard pandas `datetime64`.

## 4. Timeliness
Is the feature context statically relevant or does temporal bleed influence it?

- [ ] Extracted relevant operational timestamps.
- [ ] Addressed potential out-of-order logs.
- [ ] Addressed possible lookahead biases (e.g. counting tomorrow's stock price variance as a known feature for yesterday's prediction logic). 

## 5. Validity
Does the dataset functionally align with explicit business rules?

- [ ] Reviewed numerical extremes for logical fallacies (e.g. Employee Age < 0 or > 120). 
- [ ] Validated numeric distributions via `df.describe()`.
- [ ] Verified categorizations sit cleanly within dictionary definitions. 

## 6. Accuracy
Does the recorded metric honestly reflect the authentic, real-world context?

- [ ] Corroborated high-importance sample markers against authoritative origin contexts or alternative secondary truth tables.
- [ ] Implemented cross-checks against aggregate internal metrics. 
- [ ] Established mathematical boundary guardrails utilizing Z-Scores or IQR.
