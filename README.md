# AI-Powered Credit Risk Intelligence Platform

End-to-end credit risk platform on the Home Credit Default Risk dataset: ML risk scoring, SHAP explainability, business rule derivation, an NL-to-SQL chatbot with conversation memory + semantic caching, a Streamlit UI, and Dockerized deployment.

**Repo:** https://github.com/harinids/credit-risk-platform

## Architecture

Services (`docker-compose.yml`): `db` (Postgres 15, healthcheck-gated) → `db-init` (one-shot CSV loader) → `api` (FastAPI) + `ui` (Streamlit), started only after their dependencies report healthy/completed.

| Route | Purpose | Status |
|---|---|---|
| `GET /health` | DB connectivity | ✅ |
| `POST /chat` | NL-to-SQL + memory + cache | ✅ |
| `GET /predict/{id}` | Risk score + band | ✅ (needs raw CSVs, see Limitations) |
| `GET /explain/{id}` | SHAP explanation | ✅ (needs raw CSVs) |
| `GET /rules` | Derived business rules | ✅ no data dependency |
| `GET /eda/summary` | Dataset summary | ✅ (needs raw CSVs) |

## Setup

```bash
git clone https://github.com/harinids/credit-risk-platform.git
cd credit-risk-platform
cp .env.example .env   # fill in GROQ_API_KEY
```

Download the dataset (https://www.kaggle.com/competitions/home-credit-default-risk/data) into `data/raw/` (6 CSVs — excluded from git, ~2.68GB). Then:

```bash
docker-compose up
```

API at `:8000` (docs at `/docs`), UI at `:8501`.

Local dev without Docker (used for all NL-to-SQL iteration, via SQLite):
```bash
python -m src.nl2sql.db_setup
python -m uvicorn src.api.main:app --reload
```

## EDA — Key Findings

- 307,511 rows, 122 raw columns → 163 after relational feature engineering (34 features derived from bureau, previous_application, POS_CASH, installments, credit_card via aggregation joins)
- Class imbalance: **11.4:1** (8.07% default rate)
- `DAYS_EMPLOYED` contains a sentinel value `365243` in 55,374 rows (18%) — cleaned via NaN conversion + an explicit `days_employed_anomaly` flag rather than silently imputing
- Business insights (of 5+ generated): default rate varies materially by income type, age group, and gender; external credit bureau scores (`EXT_SOURCE_1/2/3`) are the single strongest predictors well before any model is fit
- Charts: `notebooks/eda_outputs/` (missingness, default rate by income/age/gender, SHAP global importance, SHAP waterfall)

## Machine Learning

**Model:** LightGBM. **Imbalance handling:** `scale_pos_weight` (algorithm-level), not SMOTE — avoids synthetic-sample distortion of a relational, feature-engineered dataset.

**Results:** Validation AUC-ROC = **0.7810**. At F1-optimized threshold (0.6691): Precision 0.275, Recall 0.438, F1 0.338.

**Risk bands — percentile-based**, not threshold-multiple-based:
| Band | Cutoff | Actual default rate |
|---|---|---|
| Low | bottom 60% (<0.415) | 3.0% |
| Medium | next 30% (<0.702) | 11.0% |
| High | top 10% (≥0.702) | 29.7% |

Percentile banding was chosen over naive multiples of the F1 threshold, which produced a degenerate 99.8%/0.2% split. Percentile bands give a clean, monotonically increasing risk gradient — the property that actually matters for a usable risk tier.

**Top features** (LightGBM importance and SHAP agree): `ext_source_2`, `ext_source_3`, `ext_source_1` dominate; `organization_type`, `credit_term_years`, `install_late_rate`, `bureau_debt_credit_ratio` follow.

## Explainable AI

SHAP (`TreeExplainer`) generates per-prediction factor breakdowns, converted to plain-English summaries for non-technical users (e.g. "flagged High risk — main factors: external source 2 increases risk; install late rate increases risk").

**Honest limitation surfaced during testing:** the applicant SHAP identified as highest-risk (88.9% predicted probability) actually repaid. This is a genuine false positive, consistent with the 27.5% precision at the tuned threshold — reported here rather than cherry-picking a cleaner example, since a credible risk tool should show its real failure modes.

## Business Rule Derivation

A `DecisionTreeRegressor` surrogate is fit on the LightGBM model's **predicted probabilities** (not raw labels), so rule fidelity is measured as R² against the model's own output — the rules approximate the model, not the ground truth.

| max_depth | Fidelity (R²) | Rules |
|---|---|---|
| 2 | 0.38 | 4 |
| 3 | 0.47 | 8 |
| **4 (chosen)** | **0.53** | **16** |
| 5 | 0.58 | 32 |
| 6 | 0.63 | 62 |
| 8 | 0.69 | 151 |

Depth 4 was chosen as the interpretability/fidelity trade-off point: a credit analyst can act on 16 rules; 151 rules is a decision log, not a policy. Full sensitivity table and rule text are served live at `GET /rules`.

## Talk-to-Data (NL-to-SQL) System

**Provider:** Groq free tier, `openai/gpt-oss-20b` (migrated from `llama-3.1-8b-instant` after Groq deprecated it — model name is an env var, so the swap was one line).

**Reasoning-model quirk:** `gpt-oss-20b` spends tokens on hidden chain-of-thought before visible output. `max_tokens=150-300` sometimes let hidden reasoning consume the entire budget, returning an empty visible response despite a successful API call. Fixed with `reasoning_effort="low"` + higher `max_tokens` (500 for SQL generation, 300 for the relevance check).

**Safety layers (all must pass before a query touches the database):** sqlglot parse validation → SELECT-only enforcement → table allow-list → automatic `LIMIT` injection.

**Hallucination control:** SQL is grounded in a schema retrieved specifically for the question (curated per-table keyword lists, not generic description matching — see Prompt Engineering below), then a self-consistency check asks the LLM whether the returned data actually answers the question before it's shown to the user.

### Conversation memory — bugs found and fixed

Isolated unit tests passed on the first try; **real multi-turn bugs only surfaced once schema retrieval, conversation memory, and SQL generation were tested together**:

1. **Schema retrieval scope** — `get_relevant_schema_text(question)` only considered the current question's keywords, so a table used in Turn 1 silently dropped out of scope for a Turn 2 follow-up that didn't repeat its name. Fixed by retrieving on `f"{conversation_context} {question}"`.
2. **Keyword normalization** — conversation context carries raw SQL with underscored table names (`previous_application`) but the keyword list used spaced phrases (`previous application`); the substring match never fired. Fixed by normalizing underscores to spaces before matching.
3. **Prompt didn't force query reuse** — even with the right table visible, the model sometimes invented an unrelated query for a follow-up instead of extending the prior one. Fixed with an explicit "if this is a follow-up, start from the previous SQL and modify minimally" instruction.

### Semantic caching — bug found and fixed

4. **Cache key always included conversation context** — since context accumulates turns within a session, a repeated standalone question got a different cache key each time and never hit, even for word-for-word repeats. Fixed: context is only folded into the cache key when the question actually looks like a follow-up (pronoun/continuation markers like "that", "it", "now").

**Measured cache performance:** exact-repeat speedups of **62.4x–373x** across separate test runs (typical LLM round-trip ~1.2–1.4s → cache hit ~0.00–0.02s). Reporting a range rather than a single number since LLM API latency varies call to call.

**Known, deliberately undone limitation:** cache similarity uses `difflib.SequenceMatcher` (character-level), which catches whitespace/case/rewording duplicates but misses true synonym paraphrases of different lengths — verified: "car" vs "vehicle" scored ~0.85, just under the 0.90 threshold, correctly treated as a cache miss rather than a false hit. This was a deliberate choice over embedding-based similarity: embeddings would catch the paraphrase, but at the cost of an extra API call per lookup (working against the optimization goal) and a higher risk of merging two genuinely different questions and serving a wrong cached answer — a worse failure mode for a credit-risk tool than an occasional avoidable cache miss.

## Prompt Engineering & Token Optimization

**Schema grounding:** curated per-table keyword lists (not generic description-word matching, which was too noisy — "credit" appears in nearly every table's description and caused wrong table selection). Achieves **50–66% token reduction** vs. dumping the full schema, verified per-question.

**Conversation memory scope:** capped at the last 3 turns per session (not unbounded), since every turn included in context adds tokens to every subsequent call — a direct token-cost vs. context-quality trade-off, same category of decision as the schema retrieval choice above.

## Known Limitations

- `/predict`, `/explain`, and `/eda/summary` require the raw dataset CSVs, which are excluded from the repo (~2.68GB, past what's reasonable to commit). In a fresh clone/Codespace without the dataset added to `data/raw/`, these three endpoints return a clear error rather than crashing silently; `/rules` and `/chat` work standalone since they rely on pre-computed artifacts. All three were fully exercised and verified against the complete local dataset during development.
- Conversation memory and semantic cache are in-process, in-memory stores — state is lost on API restart and does not share across multiple worker processes. Acceptable for this assignment's scope; a production deployment would back these with Redis.
- Semantic cache uses character-level similarity, not embeddings (see NL-to-SQL section above for the reasoning and a concrete example of the trade-off).
- Model precision at the tuned threshold is 27.5% — meaningful screening lift for a genuinely hard, imbalanced 8%-default-rate problem, but not accurate enough for uncontested automated approval/denial. Positioned as decision support, not decision replacement.

## Engineering Journey: Docker/WSL2 Blocker

Local Docker Desktop on Windows failed with "Virtualization support not detected," despite BIOS/Task Manager confirming virtualization was enabled. Root-cause diagnosis, in order:

1. Core Isolation / Memory Integrity — checked, already off, not the cause
2. Hyper-V hardware prerequisites (`systeminfo`) — all four passed (VM Monitor Mode, firmware virtualization, SLAT, DEP)
3. Credential Guard — checked via `Win32_DeviceGuard`, no active security services, ruled out
4. `dism /online /get-featureinfo /featurename:VirtualMachinePlatform` → **`State: Disabled`** — the actual root cause
5. Enabling it failed with **error 14098, component store corrupted**
6. `DISM /Online /Cleanup-Image /RestoreHealth` completed successfully; `sfc /scannow` found zero integrity violations
7. Re-attempting the feature enable **failed again with the identical 14098 error** — confirming the corruption sits specifically in the manifests `VirtualMachinePlatform`/WSL depend on, in a layer neither general repair tool can reach

**Decision:** rather than pursue a Windows repair-install or offline DISM servicing with matched install media (large, uncertain time cost), Docker verification was moved to GitHub Codespaces. All four services (`db`, `db-init`, `api`, `ui`) were confirmed to build, start, pass healthchecks, and respect dependency ordering there — the actual Docker/Compose engineering was verified correct; the blocker was local-machine-specific OS corruption, not the project.

## EDA — Key Findings

- 307,511 rows, 122 raw columns -> 163 after relational feature engineering (34 features derived from bureau, previous_application, POS_CASH, installments, credit_card via aggregation joins)
- Class imbalance: 11.4:1 (8.07% default rate)
- DAYS_EMPLOYED contains a sentinel value 365243 in 55,374 rows (18%) - cleaned via NaN conversion plus an explicit days_employed_anomaly flag rather than silently imputing
- Business insights (of 5+ generated): default rate varies materially by income type, age group, and gender; external credit bureau scores (EXT_SOURCE_1/2/3) are the single strongest predictors well before any model is fit
- Charts: notebooks/eda_outputs/ (missingness, default rate by income/age/gender, SHAP global importance, SHAP waterfall)

## Machine Learning

Model: LightGBM. Imbalance handling: scale_pos_weight (algorithm-level), not SMOTE - avoids synthetic-sample distortion of a relational, feature-engineered dataset.

Results: Validation AUC-ROC = 0.7810. At F1-optimized threshold (0.6691): Precision 0.275, Recall 0.438, F1 0.338.

Risk bands - percentile-based, not threshold-multiple-based:

| Band | Cutoff | Actual default rate |
|---|---|---|
| Low | bottom 60% (<0.415) | 3.0% |
| Medium | next 30% (<0.702) | 11.0% |
| High | top 10% (>=0.702) | 29.7% |

Percentile banding was chosen over naive multiples of the F1 threshold, which produced a degenerate 99.8%/0.2% split. Percentile bands give a clean, monotonically increasing risk gradient.

Top features (LightGBM importance and SHAP agree): ext_source_2, ext_source_3, ext_source_1 dominate; organization_type, credit_term_years, install_late_rate, bureau_debt_credit_ratio follow.

## Explainable AI

SHAP (TreeExplainer) generates per-prediction factor breakdowns, converted to plain-English summaries for non-technical users.

Honest limitation surfaced during testing: the applicant SHAP identified as highest-risk (88.9% predicted probability) actually repaid - a genuine false positive, consistent with the 27.5% precision at the tuned threshold. Reported here rather than cherry-picking a cleaner example.

## Business Rule Derivation

A DecisionTreeRegressor surrogate is fit on the LightGBM model's predicted probabilities (not raw labels); fidelity is R2 against the model's own output.

| max_depth | Fidelity (R2) | Rules |
|---|---|---|
| 2 | 0.38 | 4 |
| 3 | 0.47 | 8 |
| 4 (chosen) | 0.53 | 16 |
| 5 | 0.58 | 32 |
| 6 | 0.63 | 62 |
| 8 | 0.69 | 151 |

Depth 4 chosen as the interpretability/fidelity trade-off: a credit analyst can act on 16 rules; 151 rules is a decision log, not a policy. Served live at GET /rules.

## Talk-to-Data (NL-to-SQL) System

Provider: Groq free tier, openai/gpt-oss-20b (migrated from llama-3.1-8b-instant after Groq deprecated it - model name is an env var, so the swap was one line).

Reasoning-model quirk: gpt-oss-20b spends tokens on hidden chain-of-thought before visible output. Low max_tokens sometimes let hidden reasoning consume the entire budget, returning an empty visible response despite a successful API call. Fixed with reasoning_effort="low" plus higher max_tokens.

Safety layers: sqlglot parse validation, SELECT-only enforcement, table allow-list, automatic LIMIT injection.

Hallucination control: SQL is grounded in a schema retrieved specifically for the question, then a self-consistency check asks the LLM whether the returned data actually answers the question before it is shown to the user.

### Conversation memory - bugs found and fixed

Real multi-turn bugs only surfaced once schema retrieval, conversation memory, and SQL generation were tested together:

1. Schema retrieval scope - only considered the current question's keywords, so a table used in Turn 1 dropped out of scope for a Turn 2 follow-up. Fixed by retrieving on the combined context plus question.
2. Keyword normalization - conversation context carries underscored table names but the keyword list used spaced phrases. Fixed by normalizing before matching.
3. Prompt did not force query reuse - model sometimes invented an unrelated query for a follow-up. Fixed with an explicit "start from the previous SQL, modify minimally" instruction.

### Semantic caching - bug found and fixed

4. Cache key always included conversation context - a repeated standalone question got a different cache key each time as context accumulated, guaranteeing a miss even for word-for-word repeats. Fixed: context only folds into the cache key when the question looks like an actual follow-up.

Measured cache performance: exact-repeat speedups of 62.4x-373x across separate runs (typical LLM round-trip ~1.2-1.4s versus a cache hit at ~0.00-0.02s).

Known limitation, deliberately not fixed: character-level similarity (difflib) catches rewording duplicates but misses true synonym paraphrases of different lengths - verified "car" vs "vehicle" scored ~0.85, just under the 0.90 threshold. This is deliberate: embeddings would catch the paraphrase but cost an extra API call per lookup and risk merging genuinely different questions - a worse failure mode for a credit tool than an occasional avoidable miss.

## Prompt Engineering and Token Optimization

Schema grounding: curated per-table keyword lists, not generic description-word matching (too noisy - "credit" appears in nearly every table description). Achieves 50-66% token reduction versus the full schema dump, verified per-question.

Conversation memory scope: capped at the last 3 turns per session, since every included turn adds tokens to every subsequent call.

## Known Limitations

- /predict, /explain, and /eda/summary require the raw dataset CSVs, excluded from the repo (~2.68GB). Without the dataset in data/raw/, these return a clear error rather than crashing silently; /rules and /chat work standalone. All three were fully exercised against the complete local dataset during development.
- Conversation memory and semantic cache are in-process, in-memory stores; state resets on API restart and does not share across worker processes.
- Semantic cache uses character-level similarity, not embeddings (see above for the trade-off).
- Model precision at the tuned threshold is 27.5% - meaningful screening lift for a hard, imbalanced problem, but positioned as decision support, not automated decision replacement.

## Engineering Journey: Docker/WSL2 Blocker

Local Docker Desktop on Windows failed with "Virtualization support not detected," despite BIOS/Task Manager confirming virtualization enabled. Root-cause diagnosis:

1. Core Isolation/Memory Integrity checked - already off, not the cause
2. Hyper-V hardware prerequisites all passed (VM Monitor Mode, firmware virtualization, SLAT, DEP)
3. Credential Guard checked via Win32_DeviceGuard - no active security services, ruled out
4. dism /online /get-featureinfo /featurename:VirtualMachinePlatform showed State: Disabled - the actual root cause
5. Enabling it failed with error 14098, component store corrupted
6. DISM /Online /Cleanup-Image /RestoreHealth completed successfully; sfc /scannow found zero integrity violations
7. Re-attempting the feature enable failed again with the identical 14098 error, confirming the corruption sits specifically in manifests neither general repair tool can reach

Decision: rather than pursue a Windows repair-install, Docker verification was moved to GitHub Codespaces. All four services were confirmed to build, start, pass healthchecks, and respect dependency ordering there - the Docker/Compose engineering was verified correct; the blocker was local-machine-specific OS corruption, not the project.
