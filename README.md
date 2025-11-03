# 🧠 Causal Recourse Framework for Suicide Risk Reduction

> A clinically grounded and ethically aware machine learning framework for generating personalized counterfactual recommendations to reduce suicidal ideation.  
> Implementation accompanying the Master's dissertation (UFMG, 2025).

---

### 📌 Overview

This repository contains the source code and implementation of a causal recourse framework developed as part of a Master's dissertation at UFMG. The goal is to support clinical decision-making in mental health by generating actionable, plausible and individualized recommendations that help reduce suicide risk.

### Key components:
- **Causal variable selection** (NOTEARS / structure learning) to identify plausible causes.  
- **ICoNet** (counterfactual inference with a latent individuality factor) to capture patient-specific latent traits.  
- **Optimization with Simulated Annealing (SA)** to search for low-cost, plausible interventions and to incorporate ethical penalties (e.g., immutable attributes).  
- Evaluation and baseline comparisons (DiCE, Alibi, ReLax, ICR, CARLA).  

The codebase is modular so the pipeline can be adapted to other clinical or behavioral datasets.

---

### 📂 Project Structure

```
causal-recourse-mentalhealth/
│
├── baseline/ # Baseline implementations and comparison scripts
├── causal_inference/ # NOTEARS implementation and causal graph analysis
├── contrafactual_analysis/# Counterfactual generation and analysis notebooks
├── data/ # Instructions and helpers for STAR*D dataset (no raw data included)
├── IRT/ # Item Response Theory utilities
├── modeling/ # ICoNet implementation and training scripts
├── optimization/ # Simulated Annealing and optimization routines
├── preprocessing/ # Data cleaning and transformation scripts
├── results/ # Output figures, tables and metrics
├── shap/ # SHAP explanation scripts
├── requirements.txt # Python dependencies
└── README.md # This file
```

---

### ⚙️ Setup & Installation

```bash
git clone https://github.com/Cristianemax/causal-recourse-mentalhealth.git
cd causal-recourse-mentalhealth
pip install -r requirements.txt
```

Requirements include:

- Python ≥ 3.8
- PyTorch
- NetworkX
- pgmpy
- scikit-learn
- matplotlib
- pandas
- torch==1.12.1
- pytorch-lightning==1.7.7
- torchmetrics==0.10.0
- m atplotlib

---

### 🚀 How to Run

1. Prepare your dataset (default: STAR*D) following the preprocessing instructions.
2. Run the causal structure learning:
   ```bash
   causal_inference/causalnex.py
   ```
3. Train ICoNet and generate counterfactuals:
   ```bash
   python modeling/counterfactural_inference_GCN.py
   ```
4. Apply Simulated Annealing to optimize recommendations:
   ```bash
   python optimization/simulacao_contrafactual_novos_individuosSN.py
   ```

---

### 🔄 Adaptation to Other Datasets

The framework is modular and can be adapted to other clinical or behavioral datasets. To use a different dataset:

- Format your dataset with appropriate features (categorical, ordinal or binary).
- Define the causal graph or allow NOTEARS to infer it.

---

### 📊 Outputs

- Personalized counterfactual recommendations
- Causal graphs (PNG)
- Feature impact analysis
- Evaluation metrics (plausibility, cost, ethical constraints)

Evaluation metrics:

Prediction accuracy per target

PPA (Proportion of Acceptable Predictions) — tolerance-based metric

Coverage: % individuals with generated recourse

Number of reductions / increases in target

Energy trace from SA

---
🎯 Multi-Target Learning (Note & Justification)
Why multi-target?
In this study, we predict two related clinical outcomes jointly: suicidal ideation and anxiety level.
Modeling them jointly (multi-target learning) is conceptually and empirically advantageous because:

Clinical outcomes are often correlated and influenced by shared latent factors (e.g., temperament, comorbidity).

A shared latent representation (the individuality factor φ in ICoNet) captures patient-specific effects that simultaneously influence multiple responses.

Joint training allows leveraging cross-target signals, improving robustness and producing coherent counterfactuals (avoiding recommendations that reduce one target while worsening another).

Relation to ICoNet
ICoNet’s architecture is explicitly designed as a multi-target model — the encoder estimates φ from joint residuals of base models predicting multiple outcomes, and the decoder takes φ plus counterfactual predictors to output a vector of counterfactual targets.
If the model were single-target, φ would have limited ability to explain shared latent structure and could degenerate into noise, reducing personalization benefits.

---
🔐 Data Access — STAR*D (Instructions)
The STARD dataset is controlled-access. This repository does not include raw STARD data.

How to obtain STAR*D
Visit the NIMH Data Archive portal (or the data custodians' STAR*D page).

Register for an account and request access to the STAR*D dataset (may require institutional approval).

Once access is granted, download the de-identified data and place it in data/.
---
🔬 Evaluation Notes & Reproducibility
To reproduce results from the dissertation, use the provided config files and the package versions listed in requirements.txt.

Random seeds and logs are stored in results/experiments/.

Figures and tables generated by notebooks correspond directly to those referenced in the dissertation.
---

---

### 📚 Reference

This work is based on the dissertation:

> **Cristiane Máximo de Freitas**  
> *Framework de Recourse Algorítmico para Apoio à Decisão Clínica na Redução da Ideação Suicida*  
> UFMG – Programa de Pós-Graduação em Ciência da Computação (2025)

---

### 🤝 Citation

If you use this work, please cite:

```bibtex
@mastersthesis{freitas2025recourse,
  title={Framework de Recourse Algorítmico para Apoio à Decisão Clínica na Redução da Ideação Suicida},
  author={Freitas, Cristiane Máximo de},
  school={Universidade Federal de Minas Gerais},
  year={2025}
}
```
📚 References (Select)
```bibtex
@article{marchezini2022counterfactual,
  title={Counterfactual inference with latent variable and its application in mental health care},
  author={Marchezini, M. and others},
  journal={Frontiers in Artificial Intelligence},
  volume={5},
  pages={853437},
  year={2022},
  doi={10.3389/frai.2022.853437}
}

@article{kirkpatrick1983optimization,
  title={Optimization by Simulated Annealing},
  author={Kirkpatrick, Scott and Gelatt, C. D. and Vecchi, M. P.},
  journal={Science},
  volume={220},
  number={4598},
  pages={671--680},
  year={1983},
  doi={10.1126/science.220.4598.671}
}

@article{rush2004sequenced,
  title={The Sequenced Treatment Alternatives to Relieve Depression (STAR*D): rationale and design},
  author={Rush, A. J. and others},
  journal={Controlled Clinical Trials},
  volume={25},
  number={1},
  pages={119--142},
  year={2004},
  doi={10.1016/S0197-2456(03)00112-0}
}
---

### 📬 Contact

Feel free to reach out:
- Email: cristianemaximo@gmail.com
- LinkedIn: [Cristiane Freitas]([https://www.linkedin.com/in/cristian/](https://www.linkedin.com/in/cristiane-freitas-601b2016?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app))

---

### 🧾 License

This project is released under the MIT License. See [LICENSE](./LICENSE) for more information.
