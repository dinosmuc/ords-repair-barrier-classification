# Multilingual Classification of End-of-Life Repair Barriers

## Project objective

This project investigates whether the reason that a product reaches end of life can be predicted automatically from multilingual free-text community-repair descriptions, and whether adding LLM-generated training labels improves the resulting classifiers.

Two model families will be compared:

- A shallow classifier using word and character TF-IDF features with `LinearSVC`.
- A deep classifier using full end-to-end fine-tuning of [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) with a six-class classification head.

Each model will be trained once on original Open Repair Data Standard (ORDS) labels and once on the same original training data plus accepted LLM pseudo-labels. This produces a prespecified 2 × 2 design:

| System | Model family | Training labels |
| --- | --- | --- |
| LinearSVC-O | TF-IDF + LinearSVC | Original only |
| LinearSVC-E | TF-IDF + LinearSVC | Original + accepted LLM labels |
| mmBERT-O | Fine-tuned mmBERT | Original only |
| mmBERT-E | Fine-tuned mmBERT | Original + accepted LLM labels |

## Intended contribution

This project will create a reproducible benchmark and empirical evaluation for predicting the six ORDS repair-barrier categories from multilingual community-repair descriptions. It will provide duplicate-grouped data splits, baseline implementations, evaluation code, and a prespecified comparison of TF-IDF/LinearSVC and mmBERT trained with original versus LLM-enriched data. All primary comparisons will use a locked test set containing only original ORDS labels.

At the time of the structured literature search in August 2026, no published benchmark specifically addressing multilingual ORDS repair-barrier prediction was identified. This is therefore an empirical and resource contribution, not a claim to a new learning algorithm.

## Data

The project uses the July 2025 ORDS aggregate. The six target categories are:

1. Spare parts not available.
2. Spare parts too expensive.
3. No way to open the product.
4. Repair information not available.
5. Lack of equipment.
6. Item or product too worn out.

The exact mapping between source values and these canonical classes will be stored in `config.yaml`.

A preliminary audit found:

| Dataset property | Count |
| --- | ---: |
| Source records | 305,649 |
| End-of-life records | 77,709 |
| End-of-life records with non-empty problem text | 57,541 |
| Records with a valid six-class label and text | 21,821 |
| Records with text but without a valid barrier label | 35,720 |
| Unique normalised texts in the labelled corpus | 19,231 |
| Duplicate labelled rows beyond the first occurrence | 2,590 |
| Normalised-text groups containing conflicting labels | 489 groups / 2,722 rows |

All non-empty labelled descriptions will be retained. No minimum word-count threshold will be imposed; in particular, the previous seven-word rule will not be used. Text length will instead be analysed as a continuous dataset property and reported by class.

The final data audit will report:

- Class counts and proportions.
- Character, word and tokenizer-length distributions.
- Empty and missing values.
- Exact and normalised duplicate groups.
- Duplicate groups containing conflicting original labels.
- Provider and country composition.
- Estimated language composition.
- Differences between the originally labelled and unlabelled populations.

Language will be estimated with the documented [`fastText lid.176`](https://fasttext.cc/docs/en/language-identification.html) model. Its top-ranked language will be used as descriptive metadata, without a confidence cut-off. Because language identification is uncertain for very short text, these outputs will be explicitly described as estimates. Country and provider fields will be reported separately and will not be treated as substitutes for language identification.

## Research questions

**RQ1:** Does fine-tuned mmBERT outperform TF-IDF with LinearSVC when both systems are trained on original ORDS labels?

**RQ2:** Does adding accepted LLM-generated labels improve LinearSVC and mmBERT performance on a locked test set containing only original ORDS labels?

**Supporting analysis:** How do multilingual stop-word removal and stemming affect the shallow classifier?

## Data partitioning and leakage control

Text used for duplicate grouping will be normalised with Unicode NFKC normalisation, case-folding, leading and trailing whitespace removal, and internal whitespace collapsing. Every record with the same normalised description will be assigned to the same partition.

The originally labelled corpus will be divided into approximately 70% training, 10% validation and 20% test data using a grouped split that preserves class proportions as closely as the duplicate-group constraint permits. The split seed and algorithm will be fixed in configuration. Conflicting-label duplicate groups will remain intact and their frequency in each partition will be reported rather than silently resolved.

The expected test size is approximately 4,300 records. Its adequacy will be documented through evaluation-precision analysis before model testing. Split files, record identifiers and hashes will then be locked.

Only the original training partition may be used for model fitting, prompt auditing and LLM enrichment. Any unlabelled record whose normalised text matches a validation or test description will be removed from the enrichment pool. Validation and test descriptions will never be submitted to the LLM.

The original ORDS labels are operational reference labels rather than independently adjudicated gold labels. All reported performance and LLM agreement will therefore be interpreted relative to those reference labels.

## Shallow model

The shallow pipeline will combine:

- Word-level TF-IDF n-grams.
- Character-level TF-IDF n-grams.
- `LinearSVC`.
- Class weights derived from the original training partition.

For class (c), the weight will be calculated as

\[
w_c = \frac{N}{K n_c},
\]

where (N) is the number of original training examples, (K=6) is the number of classes and (n_c) is the number of original training examples in class (c). The resulting explicit weight mapping will be reused unchanged for LinearSVC-O, LinearSVC-E, mmBERT-O and mmBERT-E.

Three preprocessing variants will be compared using validation macro F1:

1. Unicode normalisation, case-folding and whitespace normalisation.
2. Basic normalisation plus language-specific stop-word removal.
3. Basic normalisation, stop-word removal and language-specific stemming.

For variants 2 and 3, the top-ranked fastText language estimate will select a supported language resource. Text in unsupported languages will receive basic normalisation only, and resource coverage will be reported. This keeps the ablation reproducible while exposing the limitations of language-specific preprocessing.

Candidate word/character n-gram ranges and the SVM regularisation parameter will be prespecified in `config.yaml` and selected only on the validation set. The configuration selected for LinearSVC-O will be reused unchanged for LinearSVC-E.

## Deep-learning model

`jhu-clsp/mmBERT-base` will be fine-tuned end to end with a six-class linear classification head. The training procedure will use:

- Weighted cross-entropy with the fixed original-training class weights.
- Minimal text preprocessing.
- Dynamic padding.
- A maximum sequence length chosen from training-only tokenizer-length statistics and documented GPU-memory constraints.
- Validation-based early stopping and checkpoint selection.
- Three predetermined random seeds: 13, 42 and 73.

The mmBERT model revision and all optimisation settings will be pinned in configuration. Hyperparameters chosen for mmBERT-O using training and validation data will be reused for mmBERT-E. Early stopping may select a different epoch because the enriched training data are different, but the stopping rule will remain identical.

For each of mmBERT-O and mmBERT-E, the primary test prediction vector will come from the seed with the highest validation macro F1. This rule will be fixed before the test set is opened. Test metrics from all three seeds will also be reported as a stability range.

## LLM enrichment

### Frozen annotation procedure

The annotator will be the official DeepSeek API model [`deepseek-v4-flash`](https://api-docs.deepseek.com/updates/). A single zero-shot, example-free prompt will be written from the ORDS class definitions and frozen before any labelled-data audit or unlabelled-data annotation.

The same prespecified inference mode and decoding configuration will be used for every record. The planned configuration is non-thinking mode with temperature 0 to minimise sampling variability. The prompt will request schema-valid JSON containing either:

- One valid barrier category and one non-empty exact supporting span copied from the raw description; or
- `INSUFFICIENT_EVIDENCE`.

A pseudo-label will be accepted only if the returned category is valid and its evidence span is an exact substring of the raw input. `INSUFFICIENT_EVIDENCE`, malformed outputs and invalid evidence spans will not enter classifier training. Transport failures may be retried with the identical request; model outputs will not be repaired using a second prompt.

The evidence rule is an extractive-format constraint, not proof that the selected evidence or category is semantically correct. Its purpose is to make the annotation traceable and mechanically auditable. No numerical confidence threshold will be used.

### Frozen-prompt audit

After the prompt is frozen, DeepSeek will also classify every unique normalised description in the original training partition without being shown its existing label. These audit outputs will not be used as classifier training labels and will not be used to revise the prompt.

The audit will report:

- Overall abstention and accepted-label rates.
- Agreement with original labels among accepted outputs.
- Per-class agreement, recall-like coverage and abstention.
- Evidence-span and output-schema compliance.
- Results by estimated language, provider and country where support is sufficient to report them meaningfully.
- Conflicting original-label duplicate groups as a separate category.

This audit measures how the frozen LLM behaves against the available operational labels. It does not convert those labels into human-adjudicated ground truth and does not guarantee equal pseudo-label quality in the unlabelled population.

### Creation of enriched training data

DeepSeek will annotate only eligible, previously unlabelled training descriptions. All schema-valid, non-abstaining outputs with an exact evidence span will be accepted. The enriched training set will contain the complete original training partition plus these accepted pseudo-labelled records.

The enrichment audit will report:

- Numbers and percentages of accepted, abstained and invalid outputs.
- Accepted pseudo-label class distribution.
- Acceptance rates by provider, country and estimated language.
- Changes relative to the original training-label distribution.
- Token use, API cost, request date, model identifier and response metadata.

The fixed model hyperparameters and original-training class weights will be reused in the enriched systems. This prevents the enriched experiment from changing the classifier configuration or class-weighting rule at the same time as the training labels.

Accepted pseudo-labels may still be selectively concentrated in easier descriptions, languages, providers or classes. The distribution audit will quantify this selection effect, and it will be treated explicitly as a limitation.

## Learning-curve analysis

Nested 25%, 50% and 100% subsets of the original training groups will be evaluated on the validation partition. The subsets will preserve class proportions as closely as possible and use the same fixed feature, weighting and training configurations.

For mmBERT, the descriptive learning curve will use the predetermined seed 42; the full three-seed design is reserved for the four primary systems. These approximately geometrically spaced fractions are prespecified computational design points, not optimised thresholds.

The learning curves provide a reference for the benefit obtained by increasing the quantity of original labelled data. They help contextualise enrichment gains, but they do not eliminate the fact that enrichment changes label source, training-set size and training-data composition together.

## Evaluation and statistical inference

The locked test set will be evaluated only after preprocessing, prompts, hyperparameters, class weights, seed-selection rules and model-selection rules have been finalised.

The report will include:

- Macro F1 as the primary metric.
- Accuracy as a secondary descriptive metric.
- Per-class precision, recall, F1 and support.
- Confusion matrices.
- Validation and test results across all three mmBERT seeds.
- Training time, inference time and model size as practical context.

Macro F1 is primary because every repair-barrier category matters and the original class distribution is imbalanced. Per-class support will always accompany per-class performance so that results for rare categories are not overinterpreted.

Pointwise 95% confidence intervals for paired macro-F1 differences will be calculated with a duplicate-group-paired bootstrap. Normalised-description groups, rather than individual rows, will be resampled with replacement, and all systems will be evaluated on the same resampled groups. The bootstrap resample count and random seed will be fixed in configuration.

Three primary differences will be estimated:

1. mmBERT-O minus LinearSVC-O, answering RQ1.
2. LinearSVC-E minus LinearSVC-O, answering RQ2 for the shallow model.
3. mmBERT-E minus mmBERT-O, answering RQ2 for the deep model.

Each interval will use the single primary prediction vector selected by the frozen validation rule. The intervals will be interpreted individually as pointwise intervals; no family-wise claim will be made from their simultaneous coverage.

An enrichment improvement will be described as supported when its estimated difference is positive and its 95% paired interval excludes zero. An interval containing zero will be reported as inconclusive, even when the raw score is higher. A negative interval will indicate degradation.

The 2 × 2 experiment estimates the net downstream effect of the complete enrichment procedure. It does not separately identify the causal effects of pseudo-label source, added sample size and changed class/provider/language composition. The learning curves and enrichment audit provide the necessary context for this limitation.

## Avoiding arbitrary “magic numbers”

Every consequential design choice will be assigned to one of the following categories:

- Derived from training data, such as class weights and tokenizer-length statistics.
- Selected using validation data, such as preprocessing and classifier hyperparameters.
- Justified through evaluation-precision analysis, such as test-set adequacy.
- Fixed by the model or task architecture, such as the six ORDS classes.
- Prespecified as a computational design choice, such as random seeds, learning-curve fractions and bootstrap repetitions.

No text-length cut-off, LLM confidence threshold, prompt, hyperparameter or model will be selected using final test performance. The numerical seed values have no substantive interpretation; their role is to make repeated training prespecified and reproducible.

## Reproducibility

The repository will record:

- The dataset download URL, retrieval date and SHA-256 checksum.
- The source-to-canonical label mapping.
- Exact split membership and split hashes.
- Package versions and random seeds.
- The Hugging Face model revision.
- The frozen prompt and its SHA-256 hash.
- The DeepSeek endpoint, requested and returned model identifiers, mode, decoding settings and request date.
- Raw LLM outputs, validation outcomes and accepted pseudo-labels.
- Model predictions used for every reported comparison.

Because a hosted API can change behind a stable model name, the request date, returned metadata, raw outputs and prompt/configuration hashes are essential parts of the reproducibility record.

## Deliverables

The project will deliver:

1. Reproducible source code and configuration.
2. Locked data-split manifests and model prediction files.
3. A frozen LLM prompt, annotation audit and pseudo-label audit.
4. Results for the four primary systems and the preprocessing ablation.
5. A concise scientific report discussing results, uncertainty and limitations.

The main report can remain within the required page limit by presenting only the essential audit statistics, method, primary results and discussion. Full configuration grids, prompts, raw audit tables and additional seed outputs will remain in the repository or appendix.

## Technical stack

- Python and Jupyter.
- pandas and NumPy.
- scikit-learn.
- PyTorch.
- Hugging Face Transformers and Datasets.
- fastText language identification.
- DeepSeek API.
- SciPy.
- Matplotlib and Seaborn.
- Git and YAML configuration files.

## Repository structure

```text
ords-repair-barrier-classification/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── pseudo_labels/
├── prompts/
│   └── repair_barrier_zero_shot.md
├── notebooks/
│   ├── 01_data_audit_and_split.ipynb
│   ├── 02_linear_svc.ipynb
│   ├── 03_llm_enrichment.ipynb
│   └── 04_mmbert_and_evaluation.ipynb
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── language_id.py
│   ├── linear_model.py
│   ├── llm_annotation.py
│   ├── deep_model.py
│   └── evaluation.py
├── outputs/
│   ├── predictions/
│   ├── llm_audit/
│   ├── tables/
│   └── figures/
└── report/
```

## Notebook structure

### `01_data_audit_and_split.ipynb`

- Load and validate the July 2025 ORDS aggregate.
- Define the six-class mapping.
- Audit missingness, class balance, lengths, duplicates, conflicts, providers, countries and estimated languages.
- Create the duplicate-grouped, approximately stratified 70/10/20 split.
- Exclude enrichment candidates matching validation/test text.
- Save and hash the locked split manifests.

### `02_linear_svc.ipynb`

- Run the three preprocessing variants on the validation set.
- Select the word/character TF-IDF and LinearSVC configuration.
- Produce the original-label learning curve.
- Train LinearSVC-O.
- Freeze the shallow-model configuration for later LinearSVC-E training.

### `03_llm_enrichment.ipynb`

- Load and hash the frozen zero-shot prompt.
- Run the frozen-prompt audit on the original training partition.
- Annotate the eligible unlabelled training pool.
- Validate the JSON schema and exact evidence spans.
- Save accepted pseudo-labels and the enrichment audit.

### `04_mmbert_and_evaluation.ipynb`

- Train mmBERT-O and mmBERT-E with seeds 13, 42 and 73.
- Train LinearSVC-E with the frozen shallow configuration.
- Select each primary mmBERT seed using validation macro F1.
- Evaluate the four systems on the locked test set.
- Produce metrics, confusion matrices, paired bootstrap intervals, tables and figures.

## References

1. **Horych et al. (2025), *The Promises and Pitfalls of LLM Annotations in Dataset Labeling*** — [Paper](https://aclanthology.org/2025.findings-naacl.75/) — Motivates evaluating synthetic labels through downstream classifiers and independent benchmark data.

2. **Pangakis and Wolken (2024), *Knowledge Distillation in Automated Annotation*** — [Paper](https://aclanthology.org/2024.nlpcss-1.9/) — Directly studies supervised classifiers trained with LLM-generated labels across multiple classification tasks.

3. **Marone et al. (2025), *mmBERT: A Modern Multilingual Encoder with Annealed Language Learning*** — [Paper](https://arxiv.org/abs/2509.06888) — Supports mmBERT as the multilingual encoder used in the deep-learning system.

4. **Chakraborty et al. (2019), *Sparse Victory*** — [Paper](https://aclanthology.org/R19-1022/) — Supports sparse TF-IDF-style representations as a strong text-classification baseline.

5. **Uysal and Günal (2014), *The Impact of Preprocessing on Text Classification*** — [Paper](https://doi.org/10.1016/j.ipm.2013.08.006) — Supports testing stop-word removal and stemming empirically instead of assuming that they help.

6. **Opitz (2024), *A Closer Look at Classification Evaluation Metrics and a Critical Reflection of Common Evaluation Practice*** — [Paper](https://aclanthology.org/2024.tacl-1.46/) — Supports transparent metric choice and reporting macro and per-class behaviour.

7. **Bey et al. (2020), *Fold-Stratified Cross-Validation for Unbiased and Privacy-Preserving Federated Learning*** — [Paper](https://doi.org/10.1093/jamia/ocaa096) — Supports keeping duplicate or related records in the same data partition to prevent leakage.

8. **Mosbach et al. (2021), *On the Stability of Fine-Tuning BERT*** — [Paper](https://openreview.net/forum?id=nzpLWnVAyah) — Supports training transformer classifiers with multiple predetermined random seeds.

9. **Viering and Loog (2023), *The Shape of Learning Curves: A Review*** — [Paper](https://doi.org/10.1109/TPAMI.2022.3220744) — Supports nested learning curves for studying the effect of training-set size.

10. **Bestgen (2022), *Please, Don’t Forget the Difference and the Confidence Interval*** — [Paper](https://arxiv.org/abs/2205.11134) — Supports paired score differences and bootstrap confidence intervals rather than raw-score ranking alone.

11. **Beleites et al. (2013), *Sample Size Planning for Classification Models*** — [Paper](https://arxiv.org/abs/1211.1323) — Supports reasoning about test-set size through evaluation precision and independent test units.

12. **Open Repair Alliance, *Open Repair Data Standard*** — [Specification](https://standard.openrepair.org/standard.html) — Defines the repair-status and repair-barrier fields that ground the task and class taxonomy.
