# AI-Driven Fraud-Resilient Inclusive Lending

This repository accompanies the research paper:

AI-Driven Fraud-Resilient Digital Financial Platforms for Inclusive Lending:
A Reproducible Graph-Based and Explainable Framework for Emerging Economies

## Motivation
Financial inclusion failures persist not due to a lack of credit models,
but due to unmanaged fraud risk at the inclusion boundary.
This framework jointly models credit risk, fraud risk,
and inclusion constraints in a single reproducible pipeline.

## Key Contributions
- Joint optimization of credit risk, fraud risk, and inclusion coverage
- Graph-based fraud detection integrated with lending decisions
- Regulator-aligned explainability using SHAP and graph explanations
- Fully reproducible Python framework using synthetic data

## Repository Structure
configs/        # Model and decision thresholds
data/           # Synthetic and anonymized datasets
src/            # Credit, fraud, and explainability modules
experiments/    # End-to-end reproducibility scripts
docker/         # Containerized execution environment

## Reproducibility
All experiments can be reproduced using synthetic data
and open-source dependencies.

pip install -r requirements.txt
python experiments/reproduce_results.py

## Scope Note
This repository provides a research and reproducibility framework.
It is not a production system.

## License
Apache License 2.0

