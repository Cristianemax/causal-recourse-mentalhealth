🧠 Causal Recourse Framework for Suicide Risk Reduction
A clinically grounded, ethically aware machine learning framework for generating personalized counterfactual recommendations to reduce suicidal ideation.

📌 Overview
This repository contains the source code and implementation of a causal recourse framework developed as part of a Master's dissertation at UFMG. The goal is to support clinical decision-making in mental health by generating actionable, plausible and individualized recommendations that help reduce suicide risk.

The proposed approach integrates:

📊 Causal variable selection (via NOTEARS)

🔍 Counterfactual reasoning using ICoNet neural architecture

🔧 Optimization with Simulated Annealing

⚖️ Ethical constraints (e.g., penalization of immutable attributes)

📂 Project Structure
graphql
Copiar
Editar
causal-recourse-mentalhealth/
│
├── data/                  # Instructions to access STAR*D dataset
├── preprocessing/         # Scripts for data cleaning and IRT transformation
├── causal_inference/      # NOTEARS implementation and causal graph analysis
├── modeling/              # ICoNet and counterfactual generation
├── optimization/          # Simulated Annealing implementation
├── notebooks/             # Jupyter Notebooks for experimentation
├── utils/                 # Helper functions and visualizations
├── results/               # Output files and evaluation metrics
└── README.md              # This file
⚙️ Setup & Installation
bash
Copiar
Editar
git clone https://github.com/Cristianemax/causal-recourse-mentalhealth.git
cd causal-recourse-mentalhealth
pip install -r requirements.txt
Requirements include:

Python ≥ 3.8

PyTorch

NetworkX

pgmpy

scikit-learn

matplotlib

pandas

🚀 How to Run
Prepare your dataset (default: STAR*D) following the preprocessing instructions.

Run the causal structure learning:

bash
Copiar
Editar
python causal_inference/build_dag.py
Train ICoNet and generate counterfactuals:

bash
Copiar
Editar
python modeling/train_iconet.py
Apply Simulated Annealing to optimize recommendations:

bash
Copiar
Editar
python optimization/sa_runner.py
All configurations can be adjusted in config.yaml.

🔄 Adaptation to Other Datasets
The framework is modular and can be adapted to other clinical or behavioral datasets. To use a different dataset:

Format your dataset with appropriate features (categorical, ordinal or binary).

Define the causal graph or allow NOTEARS to infer it.

Adjust config.yaml and preprocessing scripts.

📊 Outputs
Personalized counterfactual recommendations

Causal graphs (PDF/PNG)

Feature impact analysis

Evaluation metrics (plausibility, cost, ethical constraints)

📚 Reference
This work is based on the dissertation:

Cristiane Máximo de Freitas
Framework de Recourse Algorítmico para Apoio à Decisão Clínica na Redução da Ideação Suicida
UFMG – Programa de Pós-Graduação em Ciência da Computação (2025)

🤝 Citation
If you use this work, please cite:

bibtex
Copiar
Editar
@mastersthesis{freitas2025recourse,
  title={Framework de Recourse Algorítmico para Apoio à Decisão Clínica na Redução da Ideação Suicida},
  author={Freitas, Cristiane Máximo de},
  school={Universidade Federal de Minas Gerais},
  year={2025}
}
📬 Contact
Feel free to reach out:

Email: cristianemaximo@gmail.com

LinkedIn: Cristiane Freitas

🧾 License
This project is released under the MIT License. See LICENSE for more information.

