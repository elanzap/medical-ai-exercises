# Medical AI exercises

Runnable Python notebooks for the free [Medical AI course](https://ramkumarmd.com/medical-ai) by Dr. Ram Kumar G: ten tracks covering healthcare data, machine learning, deep learning and imaging, clinical NLP, large language models, retrieval-augmented generation, AI agents, evaluation and safety, regulation, and applied diabetes case studies.

Each notebook reproduces the lesson's worked example, then sets out the exercise steps for you to complete. Worked solutions are being added track by track.

## Running the notebooks

- **In your browser:** use the *Open in Colab* links below. You need a free Google account; nothing is installed on your computer.
- **On your computer:** install Python 3.11 or newer, then:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Some notebooks download a small open embedding model (all-MiniLM-L6-v2) the first time they run. The MCP notebook needs Python 3.10 or newer.

## Data and safety

- All data are **synthetic** (generated in the notebooks) or **openly licensed** and de-identified. Datasets that need credentialed access, such as the full MIMIC-IV, are linked with instructions and never copied here.
- **Never use real, identifiable patient data** in these notebooks, and never paste it into public AI tools.
- The code and models are for **education only**. They are not medical devices and must not be used to make decisions about real patients.

## Notebooks

### 1. Foundations

| Lesson | Notebook | Run | Solution |
|---|---|---|---|
| [Healthcare data types: records, images, signals, notes and omics](https://ramkumarmd.com/medical-ai/foundations/healthcare-data-types) | [notebook](notebooks/foundations/02-healthcare-data-types.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/foundations/02-healthcare-data-types.ipynb) | [solution](notebooks/foundations/02-healthcare-data-types-solution.ipynb) |
| [Data standards: FHIR, HL7, ICD, SNOMED CT, LOINC and OMOP](https://ramkumarmd.com/medical-ai/foundations/healthcare-data-standards) | [notebook](notebooks/foundations/03-healthcare-data-standards.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/foundations/03-healthcare-data-standards.ipynb) | [solution](notebooks/foundations/03-healthcare-data-standards-solution.ipynb) |
| [Privacy and de-identification: HIPAA, GDPR and India's DPDP Act](https://ramkumarmd.com/medical-ai/foundations/privacy-and-de-identification) | [notebook](notebooks/foundations/04-privacy-and-de-identification.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/foundations/04-privacy-and-de-identification.ipynb) | [solution](notebooks/foundations/04-privacy-and-de-identification-solution.ipynb) |
| [Setting up Python and Colab for clinical data work](https://ramkumarmd.com/medical-ai/foundations/python-setup-for-clinicians) | [notebook](notebooks/foundations/05-python-setup-for-clinicians.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/foundations/05-python-setup-for-clinicians.ipynb) | [solution](notebooks/foundations/05-python-setup-for-clinicians-solution.ipynb) |
| [Diagnostic accuracy metrics: sensitivity, specificity, PPV, ROC and calibration](https://ramkumarmd.com/medical-ai/foundations/diagnostic-accuracy-metrics) | [notebook](notebooks/foundations/06-diagnostic-accuracy-metrics.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/foundations/06-diagnostic-accuracy-metrics.ipynb) | [solution](notebooks/foundations/06-diagnostic-accuracy-metrics-solution.ipynb) |

### 2. Machine learning

| Lesson | Notebook | Run | Solution |
|---|---|---|---|
| [Prediction or causation? Framing the clinical question](https://ramkumarmd.com/medical-ai/machine-learning/prediction-vs-causal-questions) | [notebook](notebooks/machine-learning/01-prediction-vs-causal-questions.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/machine-learning/01-prediction-vs-causal-questions.ipynb) | [solution](notebooks/machine-learning/01-prediction-vs-causal-questions-solution.ipynb) |
| [Building a cohort and avoiding data leakage](https://ramkumarmd.com/medical-ai/machine-learning/cohorts-and-data-leakage) | [notebook](notebooks/machine-learning/02-cohorts-and-data-leakage.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/machine-learning/02-cohorts-and-data-leakage.ipynb) | [solution](notebooks/machine-learning/02-cohorts-and-data-leakage-solution.ipynb) |
| [A logistic-regression risk score](https://ramkumarmd.com/medical-ai/machine-learning/logistic-regression-risk-score) | [notebook](notebooks/machine-learning/03-logistic-regression-risk-score.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/machine-learning/03-logistic-regression-risk-score.ipynb) | [solution](notebooks/machine-learning/03-logistic-regression-risk-score-solution.ipynb) |
| [Gradient-boosted trees for readmission risk](https://ramkumarmd.com/medical-ai/machine-learning/gradient-boosting-readmission) | [notebook](notebooks/machine-learning/04-gradient-boosting-readmission.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/machine-learning/04-gradient-boosting-readmission.ipynb) | [solution](notebooks/machine-learning/04-gradient-boosting-readmission-solution.ipynb) |
| [Survival analysis: time-to-event models](https://ramkumarmd.com/medical-ai/machine-learning/survival-analysis) | [notebook](notebooks/machine-learning/05-survival-analysis.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/machine-learning/05-survival-analysis.ipynb) | [solution](notebooks/machine-learning/05-survival-analysis-solution.ipynb) |
| [Missing data and class imbalance](https://ramkumarmd.com/medical-ai/machine-learning/missing-data-and-imbalance) | [notebook](notebooks/machine-learning/06-missing-data-and-imbalance.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/machine-learning/06-missing-data-and-imbalance.ipynb) | [solution](notebooks/machine-learning/06-missing-data-and-imbalance-solution.ipynb) |
| [Explaining models with SHAP](https://ramkumarmd.com/medical-ai/machine-learning/explainability-shap) | [notebook](notebooks/machine-learning/07-explainability-shap.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/machine-learning/07-explainability-shap.ipynb) | [solution](notebooks/machine-learning/07-explainability-shap-solution.ipynb) |
| [Calibration and decision-curve analysis](https://ramkumarmd.com/medical-ai/machine-learning/calibration-and-decision-curves) | [notebook](notebooks/machine-learning/08-calibration-and-decision-curves.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/machine-learning/08-calibration-and-decision-curves.ipynb) | [solution](notebooks/machine-learning/08-calibration-and-decision-curves-solution.ipynb) |
| [Unsupervised learning for patient phenotypes](https://ramkumarmd.com/medical-ai/machine-learning/unsupervised-phenotyping) | [notebook](notebooks/machine-learning/09-unsupervised-phenotyping.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/machine-learning/09-unsupervised-phenotyping.ipynb) | [solution](notebooks/machine-learning/09-unsupervised-phenotyping-solution.ipynb) |

### 3. Deep learning & imaging

| Lesson | Notebook | Run | Solution |
|---|---|---|---|
| [Neural-network basics](https://ramkumarmd.com/medical-ai/deep-learning-imaging/neural-network-basics) | [notebook](notebooks/deep-learning-imaging/01-neural-network-basics.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/deep-learning-imaging/01-neural-network-basics.ipynb) | coming |
| [CNNs for chest X-ray and retinal images](https://ramkumarmd.com/medical-ai/deep-learning-imaging/cnns-medical-imaging) | [notebook](notebooks/deep-learning-imaging/02-cnns-medical-imaging.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/deep-learning-imaging/02-cnns-medical-imaging.ipynb) | coming |
| [Image segmentation with U-Net](https://ramkumarmd.com/medical-ai/deep-learning-imaging/segmentation-unet) | [notebook](notebooks/deep-learning-imaging/03-segmentation-unet.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/deep-learning-imaging/03-segmentation-unet.ipynb) | coming |
| [ECG and physiological time-series models](https://ramkumarmd.com/medical-ai/deep-learning-imaging/ecg-time-series-models) | [notebook](notebooks/deep-learning-imaging/04-ecg-time-series-models.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/deep-learning-imaging/04-ecg-time-series-models.ipynb) | coming |
| [Transfer learning and imaging foundation models](https://ramkumarmd.com/medical-ai/deep-learning-imaging/transfer-learning-imaging-foundation-models) | [notebook](notebooks/deep-learning-imaging/05-transfer-learning-imaging-foundation-models.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/deep-learning-imaging/05-transfer-learning-imaging-foundation-models.ipynb) | coming |

### 4. Clinical NLP

| Lesson | Notebook | Run | Solution |
|---|---|---|---|
| [Working with clinical notes](https://ramkumarmd.com/medical-ai/clinical-nlp/working-with-clinical-notes) | [notebook](notebooks/clinical-nlp/01-working-with-clinical-notes.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/clinical-nlp/01-working-with-clinical-notes.ipynb) | coming |
| [Named-entity recognition and clinical coding](https://ramkumarmd.com/medical-ai/clinical-nlp/clinical-ner-and-coding) | [notebook](notebooks/clinical-nlp/02-clinical-ner-and-coding.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/clinical-nlp/02-clinical-ner-and-coding.ipynb) | coming |
| [Clinical text embeddings](https://ramkumarmd.com/medical-ai/clinical-nlp/clinical-text-embeddings) | [notebook](notebooks/clinical-nlp/03-clinical-text-embeddings.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/clinical-nlp/03-clinical-text-embeddings.ipynb) | coming |
| [De-identifying clinical free text](https://ramkumarmd.com/medical-ai/clinical-nlp/de-identifying-clinical-text) | [notebook](notebooks/clinical-nlp/04-de-identifying-clinical-text.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/clinical-nlp/04-de-identifying-clinical-text.ipynb) | coming |

### 5. Generative AI & LLMs

| Lesson | Notebook | Run | Solution |
|---|---|---|---|
| [How large language models work](https://ramkumarmd.com/medical-ai/generative-ai-llms/how-llms-work) | [notebook](notebooks/generative-ai-llms/01-how-llms-work.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/generative-ai-llms/01-how-llms-work.ipynb) | [solution](notebooks/generative-ai-llms/01-how-llms-work-solution.ipynb) |
| [Prompting for clinical tasks](https://ramkumarmd.com/medical-ai/generative-ai-llms/prompting-for-clinical-tasks) | [notebook](notebooks/generative-ai-llms/02-prompting-for-clinical-tasks.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/generative-ai-llms/02-prompting-for-clinical-tasks.ipynb) | [solution](notebooks/generative-ai-llms/02-prompting-for-clinical-tasks-solution.ipynb) |
| [Medical LLMs and their benchmarks](https://ramkumarmd.com/medical-ai/generative-ai-llms/medical-llms-and-benchmarks) | [notebook](notebooks/generative-ai-llms/03-medical-llms-and-benchmarks.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/generative-ai-llms/03-medical-llms-and-benchmarks.ipynb) | [solution](notebooks/generative-ai-llms/03-medical-llms-and-benchmarks-solution.ipynb) |
| [Hallucination and grounding](https://ramkumarmd.com/medical-ai/generative-ai-llms/hallucination-and-grounding) | [notebook](notebooks/generative-ai-llms/04-hallucination-and-grounding.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/generative-ai-llms/04-hallucination-and-grounding.ipynb) | [solution](notebooks/generative-ai-llms/04-hallucination-and-grounding-solution.ipynb) |
| [Fine-tuning with LoRA on synthetic data](https://ramkumarmd.com/medical-ai/generative-ai-llms/fine-tuning-lora) | [notebook](notebooks/generative-ai-llms/05-fine-tuning-lora.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/generative-ai-llms/05-fine-tuning-lora.ipynb) | [solution](notebooks/generative-ai-llms/05-fine-tuning-lora-solution.ipynb) |
| [Multimodal models: text, images and more](https://ramkumarmd.com/medical-ai/generative-ai-llms/multimodal-models) | [notebook](notebooks/generative-ai-llms/06-multimodal-models.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/generative-ai-llms/06-multimodal-models.ipynb) | [solution](notebooks/generative-ai-llms/06-multimodal-models-solution.ipynb) |
| [Summarisation and ambient clinical documentation](https://ramkumarmd.com/medical-ai/generative-ai-llms/summarisation-ambient-documentation) | [notebook](notebooks/generative-ai-llms/07-summarisation-ambient-documentation.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/generative-ai-llms/07-summarisation-ambient-documentation.ipynb) | [solution](notebooks/generative-ai-llms/07-summarisation-ambient-documentation-solution.ipynb) |

### 6. Retrieval-augmented generation (RAG)

| Lesson | Notebook | Run | Solution |
|---|---|---|---|
| [Why RAG? Grounding answers in trusted sources](https://ramkumarmd.com/medical-ai/rag/why-rag) | [notebook](notebooks/rag/01-why-rag.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/rag/01-why-rag.ipynb) | [solution](notebooks/rag/01-why-rag-solution.ipynb) |
| [Chunking clinical guidelines](https://ramkumarmd.com/medical-ai/rag/chunking-clinical-guidelines) | [notebook](notebooks/rag/02-chunking-clinical-guidelines.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/rag/02-chunking-clinical-guidelines.ipynb) | [solution](notebooks/rag/02-chunking-clinical-guidelines-solution.ipynb) |
| [Embeddings and vector stores](https://ramkumarmd.com/medical-ai/rag/embeddings-and-vector-stores) | [notebook](notebooks/rag/03-embeddings-and-vector-stores.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/rag/03-embeddings-and-vector-stores.ipynb) | [solution](notebooks/rag/03-embeddings-and-vector-stores-solution.ipynb) |
| [Evaluating retrieval](https://ramkumarmd.com/medical-ai/rag/retrieval-evaluation) | [notebook](notebooks/rag/04-retrieval-evaluation.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/rag/04-retrieval-evaluation.ipynb) | [solution](notebooks/rag/04-retrieval-evaluation-solution.ipynb) |
| [Citations and answer grounding](https://ramkumarmd.com/medical-ai/rag/citations-and-grounding) | [notebook](notebooks/rag/05-citations-and-grounding.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/rag/05-citations-and-grounding.ipynb) | [solution](notebooks/rag/05-citations-and-grounding-solution.ipynb) |
| [Build a guideline Q&A assistant](https://ramkumarmd.com/medical-ai/rag/guideline-qa-assistant) | [notebook](notebooks/rag/06-guideline-qa-assistant.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/rag/06-guideline-qa-assistant.ipynb) | [solution](notebooks/rag/06-guideline-qa-assistant-solution.ipynb) |
| [End-to-end RAG evaluation](https://ramkumarmd.com/medical-ai/rag/end-to-end-rag-evaluation) | [notebook](notebooks/rag/07-end-to-end-rag-evaluation.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/rag/07-end-to-end-rag-evaluation.ipynb) | [solution](notebooks/rag/07-end-to-end-rag-evaluation-solution.ipynb) |

### 7. Agentic AI

| Lesson | Notebook | Run | Solution |
|---|---|---|---|
| [Agents and tool use](https://ramkumarmd.com/medical-ai/agentic-ai/agents-and-tool-use) | [notebook](notebooks/agentic-ai/01-agents-and-tool-use.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/agentic-ai/01-agents-and-tool-use.ipynb) | [solution](notebooks/agentic-ai/01-agents-and-tool-use-solution.ipynb) |
| [Tool calling against FHIR data](https://ramkumarmd.com/medical-ai/agentic-ai/fhir-tool-calling) | [notebook](notebooks/agentic-ai/02-fhir-tool-calling.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/agentic-ai/02-fhir-tool-calling.ipynb) | [solution](notebooks/agentic-ai/02-fhir-tool-calling-solution.ipynb) |
| [Multi-agent clinical workflows](https://ramkumarmd.com/medical-ai/agentic-ai/multi-agent-clinical-workflows) | [notebook](notebooks/agentic-ai/03-multi-agent-clinical-workflows.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/agentic-ai/03-multi-agent-clinical-workflows.ipynb) | [solution](notebooks/agentic-ai/03-multi-agent-clinical-workflows-solution.ipynb) |
| [MCP and tool servers](https://ramkumarmd.com/medical-ai/agentic-ai/mcp-tool-servers) | [notebook](notebooks/agentic-ai/04-mcp-tool-servers.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/agentic-ai/04-mcp-tool-servers.ipynb) | [solution](notebooks/agentic-ai/04-mcp-tool-servers-solution.ipynb) |
| [Guardrails and human-in-the-loop design](https://ramkumarmd.com/medical-ai/agentic-ai/guardrails-human-in-the-loop) | [notebook](notebooks/agentic-ai/05-guardrails-human-in-the-loop.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/agentic-ai/05-guardrails-human-in-the-loop.ipynb) | [solution](notebooks/agentic-ai/05-guardrails-human-in-the-loop-solution.ipynb) |
| [Evaluating agents](https://ramkumarmd.com/medical-ai/agentic-ai/evaluating-agents) | [notebook](notebooks/agentic-ai/06-evaluating-agents.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/agentic-ai/06-evaluating-agents.ipynb) | [solution](notebooks/agentic-ai/06-evaluating-agents-solution.ipynb) |

### 8. Evaluation, safety & deployment

| Lesson | Notebook | Run | Solution |
|---|---|---|---|
| [External validation and dataset shift](https://ramkumarmd.com/medical-ai/evaluation-safety/external-validation-dataset-shift) | [notebook](notebooks/evaluation-safety/01-external-validation-dataset-shift.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/evaluation-safety/01-external-validation-dataset-shift.ipynb) | coming |
| [Bias and fairness](https://ramkumarmd.com/medical-ai/evaluation-safety/bias-and-fairness) | [notebook](notebooks/evaluation-safety/02-bias-and-fairness.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/evaluation-safety/02-bias-and-fairness.ipynb) | coming |
| [Reporting guidelines: TRIPOD+AI, CONSORT-AI, SPIRIT-AI, DECIDE-AI, CLAIM and more](https://ramkumarmd.com/medical-ai/evaluation-safety/reporting-guidelines) | [notebook](notebooks/evaluation-safety/03-reporting-guidelines.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/evaluation-safety/03-reporting-guidelines.ipynb) | coming |
| [Red-teaming LLM applications](https://ramkumarmd.com/medical-ai/evaluation-safety/red-teaming-llms) | [notebook](notebooks/evaluation-safety/04-red-teaming-llms.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/evaluation-safety/04-red-teaming-llms.ipynb) | coming |
| [Monitoring and MLOps](https://ramkumarmd.com/medical-ai/evaluation-safety/monitoring-and-mlops) | [notebook](notebooks/evaluation-safety/05-monitoring-and-mlops.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/evaluation-safety/05-monitoring-and-mlops.ipynb) | coming |

### 9. Regulation, ethics & governance

| Lesson | Notebook | Run | Solution |
|---|---|---|---|
| [Software as a Medical Device (SaMD)](https://ramkumarmd.com/medical-ai/regulation-ethics/software-as-a-medical-device) | [notebook](notebooks/regulation-ethics/01-software-as-a-medical-device.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/regulation-ethics/01-software-as-a-medical-device.ipynb) | coming |
| [US FDA: AI-enabled devices and change-control plans](https://ramkumarmd.com/medical-ai/regulation-ethics/us-fda-ai-devices) | [notebook](notebooks/regulation-ethics/02-us-fda-ai-devices.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/regulation-ethics/02-us-fda-ai-devices.ipynb) | coming |
| [The EU AI Act and medical devices](https://ramkumarmd.com/medical-ai/regulation-ethics/eu-ai-act) | [notebook](notebooks/regulation-ethics/03-eu-ai-act.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/regulation-ethics/03-eu-ai-act.ipynb) | coming |
| [UK MHRA: AI as a medical device](https://ramkumarmd.com/medical-ai/regulation-ethics/uk-mhra-ai) | [notebook](notebooks/regulation-ethics/04-uk-mhra-ai.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/regulation-ethics/04-uk-mhra-ai.ipynb) | coming |
| [India: CDSCO and medical-device software](https://ramkumarmd.com/medical-ai/regulation-ethics/india-cdsco-ai) | [notebook](notebooks/regulation-ethics/05-india-cdsco-ai.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/regulation-ethics/05-india-cdsco-ai.ipynb) | coming |
| [WHO guidance on AI ethics and large multimodal models](https://ramkumarmd.com/medical-ai/regulation-ethics/who-guidance-ethics) | [notebook](notebooks/regulation-ethics/06-who-guidance-ethics.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/regulation-ethics/06-who-guidance-ethics.ipynb) | coming |
| [Liability and consent](https://ramkumarmd.com/medical-ai/regulation-ethics/liability-and-consent) | [notebook](notebooks/regulation-ethics/07-liability-and-consent.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/regulation-ethics/07-liability-and-consent.ipynb) | coming |

### 10. Applied case studies: diabetes & metabolic health

| Lesson | Notebook | Run | Solution |
|---|---|---|---|
| [Forecasting glucose from CGM data](https://ramkumarmd.com/medical-ai/applied-diabetes/cgm-glucose-forecasting) | [notebook](notebooks/applied-diabetes/01-cgm-glucose-forecasting.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/applied-diabetes/01-cgm-glucose-forecasting.ipynb) | coming |
| [Diabetic retinopathy screening](https://ramkumarmd.com/medical-ai/applied-diabetes/diabetic-retinopathy-screening) | [notebook](notebooks/applied-diabetes/02-diabetic-retinopathy-screening.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/applied-diabetes/02-diabetic-retinopathy-screening.ipynb) | coming |
| [Diabetic kidney disease risk modelling](https://ramkumarmd.com/medical-ai/applied-diabetes/diabetic-kidney-disease-risk) | [notebook](notebooks/applied-diabetes/03-diabetic-kidney-disease-risk.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/applied-diabetes/03-diabetic-kidney-disease-risk.ipynb) | coming |
| [Diabetic foot-ulcer image classification](https://ramkumarmd.com/medical-ai/applied-diabetes/foot-ulcer-image-classification) | [notebook](notebooks/applied-diabetes/04-foot-ulcer-image-classification.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/applied-diabetes/04-foot-ulcer-image-classification.ipynb) | coming |
| [A RAG assistant over a diabetes guideline](https://ramkumarmd.com/medical-ai/applied-diabetes/diabetes-guideline-rag-assistant) | [notebook](notebooks/applied-diabetes/05-diabetes-guideline-rag-assistant.ipynb) | [Open in Colab](https://colab.research.google.com/github/elanzap/medical-ai-exercises/blob/main/notebooks/applied-diabetes/05-diabetes-guideline-rag-assistant.ipynb) | coming |

## Licences

- Code: [MIT](LICENSE).
- Text in the notebooks and this README: [CC BY 4.0](LICENSE-CONTENT.md).

The notebooks are generated from the published lessons, so they stay in step with the course.
