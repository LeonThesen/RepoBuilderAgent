# Dataset Analysis Report

## Results Analysis
- Processed result files: 1/2
- Schema/error files: 1
- YAML parse errors: 0

### Top Programming Languages
- Rust: 1
- C: 1
- Shell: 1
- JavaScript: 1

### Top Build Tools
- cargo: 1
- clang: 1

## Token Analysis
- Processed metric files: 1/1
- YAML/schema errors: 0

### Repositories With The Highest Two-Step Token Savings
This table answers: for which repositories does the full two-step pipeline (step1 pre-filter + step2 final classification) save the most tokens vs one-shot baseline?
Savings are computed as baseline - full two-step pipeline total; negative values mean the two-step pipeline is larger than baseline.

| Repo | One-Shot Baseline | Step 1 | Step 2 | Full Two-Prompt Pipeline | Savings vs Baseline | Savings % |
|---|---:|---:|---:|---:|---:|---:|
| cyclang | 7,581 | 0 | 7,581 | 7,581 | 0 | 0.000% |

### Prompt Comparison
- One-shot baseline full prompt tokens: 7,581
- Step 1 LLM pre-filter prompt tokens: 0
- Final classification prompt tokens after LLM pre-filtering: 7,581
- Full sequential two-prompt pipeline tokens: 7,581
- Aggregate savings: final_prompt_vs_baseline=0, two_prompt_pipeline_vs_baseline=0
- Token formula: two-step total = step1 pre-filter prompt + step2 final classification prompt.

### Pipeline Flow
- Workflow definition: baseline = one-shot full classification prompt; step1 = LLM-based pre-filter that selects likely relevant files from the structure summary; step2 = final classification prompt using only those selected files; two-step total = step1 + step2.
1. Build a structure summary for the repository.
2. Step 1 prompt asks the LLM to pre-filter and select likely relevant files.
3. Gather contents only for those selected files.
4. Step 2 prompt performs final installation/classification analysis on that reduced context.
5. Compare this sequential pipeline against the one-shot baseline full prompt.

## Selected Files Analysis
- Processed selected-files docs: 1/1
- YAML/schema errors: 0

### Top File Names

### Top File Extensions
