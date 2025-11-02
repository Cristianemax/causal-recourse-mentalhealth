# 🧠 Causal Recourse Framework for Suicide Risk Reduction

> A clinically grounded, ethically aware machine learning framework for generating personalized counterfactual recommendations to reduce suicidal ideation.

---

### 📌 Overview

This repository contains the source code and implementation of a causal recourse framework developed as part of a Master's dissertation at UFMG. The goal is to support clinical decision-making in mental health by generating actionable, plausible and individualized recommendations that help reduce suicide risk.

The proposed approach integrates:

- 📊 Causal variable selection (via NOTEARS https://causalnex.readthedocs.io/en/0.4.2/causalnex.structure.notears.html)
- 🔍 Counterfactual reasoning using ICoNet (Counterfactual inference with latent variable and its application in mental health care https://pubmed.ncbi.nlm.nih.gov/35125931/)
- 🔧 Optimization with Simulated Annealing
- ⚖️ Ethical constraints (e.g., penalization of immutable attributes)

---

### 📂 Project Structure

```
causal-recourse-mentalhealth/
│
├── baseline/              # Comparison baselines
├── causal_inference/      # NOTEARS implementation and causal graph analysis
├── contrafactual_analise/ # Counterfactual analysis
├── data/                  # Instructions to access STAR*D dataset
├── IRT/ 				   # Item Response Theory
├── modeling/              # ICoNet and counterfactual generation
├── optimization/          # Simulated Annealing implementation
├── preprocessing/         # Scripts for data cleaning and IRT transformation
├── results/               # Output files and evaluation metrics
├── shap/             	   # SHapley Additive exPlanations
└── README.md              # This file
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
- Causal graphs (PDF/PNG)
- Feature impact analysis
- Evaluation metrics (plausibility, cost, ethical constraints)

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

---

### 📬 Contact

Feel free to reach out:
- Email: cristianemaximo@gmail.com
- LinkedIn: [Cristiane Freitas]([https://www.linkedin.com/in/cristian/](https://www.linkedin.com/in/cristiane-freitas-601b2016?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app))

---

### 🧾 License

This project is released under the MIT License. See [LICENSE](./LICENSE) for more information.
