import torch
import pandas as pd
import numpy as np
import dice_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics import Accuracy

from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import train_test_split

import importlib
if importlib.util.find_spec('ipywidgets') is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm
    
dataframe = pd.read_csv('C:\\Users\Cristiane\Documents\\UFMG\Arquivos_Dissertacao\Projeto\data\\final_novo.csv', sep=';')
print(dataframe.head())

# all features
#selected = ['Suicidio','sexo', 'Estado_civil', 'Tipo_Resid','idade',
#                  'Alcoolatra', 'Droga', 'Suic_familia', 'Dep_familia',
#                   'Alc_familia', 'Drog_familia',
#                   'Neuro', 'psiquiatrica', 'Anos educacao formal', 'Capaz de desfrutar das coisas',
#                   'Impacto de sua familia e amigos',
#                   'Capaz de tomar decisões importantes', 'Estudante',
#                   'Insonia',
#                 'Deprimido', 'Ansiedade',
#                 'Perda de insights', 'Apetite', 'Perda de peso', 'Ansiedade somática',
#                  'Hipocondriase', 'Sentimentos_culpa',
#                   'Trabalho e interesses', 'Energia', 'Lentidao pensamento e fala',
#                   'Agitação', 'Libido', 'TOC']

# causal graph: 8 features
selected = [ "Suic_familia",
    "Capaz de tomar decisões importantes",
    "Estudante",
    "Hipocondriase",
    "Sentimentos_culpa",
    "Trabalho e interesses",
    'Dep_familia',
    'Alc_familia',
    'Capaz de desfrutar das coisas',
    'Droga',
    'Suicidio'#,
   #'Ansiedade'
            ]

dataframe['sexo'].replace({'M': 0, 'F': 1}, inplace=True)
dataframe['sexo'].fillna(0, inplace=True) 

df_suic = dataframe[selected]

df_suic.dropna(inplace=True)
df_suic = df_suic.astype(int)


class MyDataset(Dataset):
    def __init__(self, input_dataframe, split="train", target="Suicidio", ignore_columns=[], train_ratio=0.8):
        self.split = split
        self.target = target
        self.ignore_columns = ignore_columns

        # Remover colunas que devem ser ignoradas
        for coll in self.ignore_columns:
            if coll in input_dataframe.columns:
                input_dataframe = input_dataframe.drop(coll, axis=1)

        # Preparar as dimensões
        self.classification_dim = len(input_dataframe[self.target].unique())
        self.data_dim = len(input_dataframe.columns) - 1  # Total de features sem o alvo

        # Separar alvo e características
        y = input_dataframe[target].values
        x = input_dataframe.drop(target, axis=1).values

        # Dividir em conjunto de treino e teste
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=1 - train_ratio,
                                                                                random_state=42)

    def __len__(self):
        if self.split == "train":
            return len(self.x_train)
        elif self.split == "test":
            return len(self.x_test)
        else:
            raise ValueError("Split must be train or test")

    def __getitem__(self, idx):
        target = torch.zeros(self.classification_dim)  # Usar um tensor zero para o alvo
        if self.split == "train":
            target[self.y_train[idx]] = 1  # Definir o target correspondente para a classe
            return (torch.tensor(self.x_train[idx], dtype=torch.float32), target)  # Converter para float32
        elif self.split == "test":
            target[self.y_test[idx]] = 1  # O mesmo para o conjunto de teste
            return (torch.tensor(self.x_test[idx], dtype=torch.float32), target)  # Converter para float32
        else:
            raise ValueError("Split must be train or test")


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_layers=2):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Dropout(0.5))
            self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_layers=2):
        super(ClassificationModel, self).__init__()
        self.mlp = MLP(input_dim, output_dim, hidden_dim, n_layers)

    def forward(self, x):
        x = self.mlp(x)
        return F.softmax(x, dim=1)


class BaseModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.model = ClassificationModel(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
        self.lr = 1e-3
        self.save_hyperparameters()
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()  # Saída do modelo
        loss = F.binary_cross_entropy(y_hat, y)  # Use a função de perda apropriada para o seu problema
        acc = self.accuracy(y_hat, y.int())
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.binary_cross_entropy(y_hat, y)  # Use a função de perda apropriada para o seu problema
        acc = self.accuracy(y_hat, y.int())
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# Prepare seus dados e instancie o dataset
train_dataset = MyDataset(df_suic, split="train", target="Suicidio", ignore_columns=[], train_ratio=0.8)
test_dataset = MyDataset(df_suic, split="test", target="Suicidio", ignore_columns=[], train_ratio=0.2)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # Shuffle para treino
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Initialize model
model = BaseModel(input_dim=train_dataset.data_dim, output_dim=train_dataset.classification_dim, hidden_dim=128)

# Initialize callbacks
checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints/', filename='best-checkpoint',
                                      save_top_k=1, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.05, patience=10, mode='min')

# Initialize a trainer
trainer = pl.Trainer(
    devices=1,
    check_val_every_n_epoch=10,
    log_every_n_steps=10,
    callbacks=[checkpoint_callback, early_stopping],
)

# Train the model
trainer.fit(model, train_loader, test_loader)

# Prepare para DICE
d = dice_ml.Data(dataframe=df_suic, continuous_features=[
    "Suic_familia", "Capaz de tomar decisões importantes", "Estudante",
    "Hipocondriase", "Sentimentos_culpa", "Trabalho e interesses",
    'Dep_familia', 'Alc_familia', 'Capaz de desfrutar das coisas', 'Droga'
], outcome_name='Suicidio')

# Carregar o modelo
m = dice_ml.Model(model=model, backend='PYT')
exp = dice_ml.Dice(d, m)
#########
'''df_suic['Id'] = df_suic.index
k_0 = df_suic.loc[df_suic['Suicidio'] != 0]
k_1 = df_suic.loc[df_suic['Suicidio'] != 1]
k_2 = df_suic.loc[df_suic['Suicidio'] != 2]
k_3 = df_suic.loc[df_suic['Suicidio'] != 3]
k_4 = df_suic.loc[df_suic['Suicidio'] != 4]
k_0 = k_0.drop(columns='Suicidio')
k_1 = k_1.drop(columns='Suicidio')
k_2 = k_2.drop(columns='Suicidio')
k_3 = k_3.drop(columns='Suicidio')
k_4 = k_4.drop(columns='Suicidio')'''

#########
# Criar um sample para geração de contra-factuais
df_suic['Id'] = df_suic.index
k = df_suic.drop(columns='Suicidio')
k_4 = df_suic.loc[df_suic['Suicidio'] != 4]
k_4 = k_4.drop(columns='Suicidio')
#k_tensor = torch.tensor(k.values, dtype=torch.float32)  # Certifique-se de que está no formato correto

# Função para gerar contrafactuais com parâmetros ajustáveis
def generate_counterfactuals_for_class(k, desired_class, total_CFs=3):#, proximity_weight=0.1, diversity_weight=1.0, stopping_threshold=0.6, sparsity_weight=0.2):
    # Remover a coluna alvo antes de gerar contrafactuais
    all_counterfactuals = []
    # Gerar os contrafactuais com parâmetros ajustados
    exp_g =  exp.generate_counterfactuals(k.drop(columns='Id'),
        total_CFs=total_CFs,
        desired_class=desired_class,
        #proximity_weight=proximity_weight,
       #diversity_weight=diversity_weight,
       #stopping_threshold=stopping_threshold,
        #sparsity_weight =sparsity_weight
    )

    # Coletar os contrafactuais gerados para cada exemplo
    for i, cf_example in enumerate(exp_g.cf_examples_list):
        if cf_example and cf_example.final_cfs_df is not None:
            cf_example.final_cfs_df['original_index'] = k.loc[k.index[i], 'Id']
            all_counterfactuals.append(cf_example.final_cfs_df)
        else:
            print(f'Não foi possível gerar contrafactuais para o exemplo: {i}')

    # Concatenar os resultados em um único dataframe, se houver contrafactuais gerados
    if all_counterfactuals:
        concatenate_df = pd.concat(all_counterfactuals)
        return concatenate_df
    else:
        return pd.DataFrame()  # Retorna um dataframe vazio caso nenhum contrafactual seja gerado


# Exemplos de uso para gerar contrafactuais para diferentes classes
'''output_0 = generate_counterfactuals_for_class(k_4, desired_class=4, total_CFs=36)
if not output_0.empty:
    output_0.to_csv('C:\\Users\\Cristiane\\Documents\\UFMG\\Arquivos_Dissertacao\\Projeto\\data\\novo\\counterfactuals4_0.csv', index=False)
output_1 = generate_counterfactuals_for_class(k, desired_class=4, total_CFs=36,diversity_weight=1.5)
if not output_1.empty:
    output_1.to_csv('C:\\Users\\Cristiane\\Documents\\UFMG\\Arquivos_Dissertacao\\Projeto\\data\\novo\\counterfactuals4_.csv', index=False)
output_2 = generate_counterfactuals_for_class(k, desired_class=4, total_CFs=36,stopping_threshold=0.6)
if not output_2.empty:
    output_2.to_csv('C:\\Users\\Cristiane\\Documents\\UFMG\\Arquivos_Dissertacao\\Projeto\\data\\novo\\counterfactuals4_2.csv', index=False)
output_3 = generate_counterfactuals_for_class(k, desired_class=4, total_CFs=36,stopping_threshold=0.7)
if not output_3.empty:
    output_3.to_csv('C:\\Users\\Cristiane\\Documents\\UFMG\\Arquivos_Dissertacao\\Projeto\\data\\novo\\counterfactuals4_3.csv', index=False)
output_4 = generate_counterfactuals_for_class(k, desired_class=4, total_CFs=36, diversity_weight=2)
if not output_4.empty:
    output_4.to_csv('C:\\Users\\Cristiane\\Documents\\UFMG\\Arquivos_Dissertacao\\Projeto\\data\\novo\\counterfactuals4_4.csv', index=False)'''



# Exemplos de uso para gerar contrafactuais para diferentes classes)
output_0 = generate_counterfactuals_for_class(k, desired_class=0, total_CFs=2)#,proximity_weight=0.2)
output_1 = generate_counterfactuals_for_class(k, desired_class=1, total_CFs=2)#,proximity_weight=0.2)
output_2 = generate_counterfactuals_for_class(k, desired_class=2, total_CFs=2)#,proximity_weight=0.2)
output_3 = generate_counterfactuals_for_class(k, desired_class=3, total_CFs=2)#,proximity_weight=0.2)


# Salvar os contrafactuais gerados em arquivos CSV
if not output_0.empty:
    output_0.to_csv('C:\\Users\\Cristiane\\Documents\\UFMG\\Arquivos_Dissertacao\\Projeto\\data\\novo\\counterfactuals0.csv', index=False)
if not output_1.empty:
    output_1.to_csv('C:\\Users\\Cristiane\\Documents\\UFMG\\Arquivos_Dissertacao\\Projeto\\data\\novo\\counterfactuals1.csv', index=False)
if not output_2.empty:
    output_2.to_csv('C:\\Users\\Cristiane\\Documents\\UFMG\\Arquivos_Dissertacao\\Projeto\\data\\novo\\counterfactuals2.csv', index=False)
if not output_3.empty:
    output_3.to_csv('C:\\Users\\Cristiane\\Documents\\UFMG\\Arquivos_Dissertacao\\Projeto\\data\\novo\\counterfactuals3.csv', index=False)


'''output_0 = generate_counterfactuals_for_class(k, desired_class=0, total_CFs=3)
output_1 = generate_counterfactuals_for_class(k, desired_class=1, total_CFs=3)
output_2 = generate_counterfactuals_for_class(k, desired_class=2, total_CFs=3)
output_3 = generate_counterfactuals_for_class(k, desired_class=3, total_CFs=3)
output_4 = generate_counterfactuals_for_class(k, desired_class=4, total_CFs=10,stopping_threshold=0.7)

# Salvar os contrafactuais gerados em arquivos CSV
if not output_0.empty:
    output_0.to_csv('C:\\Users\\Cristiane\\Documents\\UFMG\\Arquivos_Dissertacao\\Projeto\\data\\novo\\counterfactuals0.csv', index=False)
if not output_1.empty:
    output_1.to_csv('C:\\Users\\Cristiane\\Documents\\UFMG\\Arquivos_Dissertacao\\Projeto\\data\\novo\\counterfactuals1.csv', index=False)
if not output_2.empty:
    output_2.to_csv('C:\\Users\\Cristiane\\Documents\\UFMG\\Arquivos_Dissertacao\\Projeto\\data\\novo\\counterfactuals2.csv', index=False)
if not output_3.empty:
    output_3.to_csv('C:\\Users\\Cristiane\\Documents\\UFMG\\Arquivos_Dissertacao\\Projeto\\data\\novo\\counterfactuals3.csv', index=False)
if not output_4.empty:
    output_4.to_csv('C:\\Users\\Cristiane\\Documents\\UFMG\\Arquivos_Dissertacao\\Projeto\\data\\novo\\counterfactuals4.csv', index=False)

'''

# Gerando os contra-factuais

'''exp_g = exp.generate_counterfactuals(k, total_CFs=3, desired_class=0, proximity_weight=0.1,  # Optional: Set proximity weight
    diversity_weight=1.0 )
print(f'verifica tamanho da lista de exemplos contrafactuais: {len(exp_g.cf_examples_list)}')

all_counterfactuals = []

for i, cf_example in enumerate(exp_g.cf_examples_list):
        if cf_example and cf_example.final_cfs_df is not None:
            cf_example.final_cfs_df['original_index'] = k.loc[k.index[i], 'Id']
            all_counterfactuals.append(cf_example.final_cfs_df)
        else:
            print(f'não foi possível gerar os contrafactuais para o exemplo: ')

if all_counterfactuals:
    concatenate_df = pd.concat(all_counterfactuals)
    concatenate_df.to_csv('C:\\Users\Cristiane\Documents\\UFMG\Arquivos_Dissertacao\Projeto\data\\novo\\counterfactuals0.csv',index=True)


exp_g = exp.generate_counterfactuals(k, total_CFs=3, desired_class=1, proximity_weight=0.1,  # Optional: Set proximity weight
    diversity_weight=1.0 )
all_counterfactuals = []

for i, cf_example in enumerate(exp_g.cf_examples_list):
        if cf_example and cf_example.final_cfs_df is not None:
            cf_example.final_cfs_df['original_index'] = k.loc[k.index[i], 'Id']
            all_counterfactuals.append(cf_example.final_cfs_df)
        else:
            print(f'não foi possível gerar os contrafactuais para o exemplo: ')

if all_counterfactuals:
    concatenate_df = pd.concat(all_counterfactuals)
    concatenate_df.to_csv('C:\\Users\Cristiane\Documents\\UFMG\Arquivos_Dissertacao\Projeto\data\\novo\\counterfactuals1.csv',index=True)


exp_g = exp.generate_counterfactuals(k, total_CFs=3, desired_class=2, proximity_weight=0.1,  # Optional: Set proximity weight
    diversity_weight=1.0 )
all_counterfactuals = []

for i, cf_example in enumerate(exp_g.cf_examples_list):
        if cf_example and cf_example.final_cfs_df is not None:
            cf_example.final_cfs_df['original_index'] = k.loc[k.index[i], 'Id']
            all_counterfactuals.append(cf_example.final_cfs_df)
        else:
            print(f'não foi possível gerar os contrafactuais para o exemplo: ')

if all_counterfactuals:
    concatenate_df = pd.concat(all_counterfactuals)
    concatenate_df.to_csv('C:\\Users\Cristiane\Documents\\UFMG\Arquivos_Dissertacao\Projeto\data\\novo\\counterfactuals2.csv',index=True)

exp_g = exp.generate_counterfactuals(k.drop(columns='Id'), total_CFs=3, desired_class=3, proximity_weight=20,  # Optional: Set proximity weight
    diversity_weight=1.0 )
all_counterfactuals = []

for i, cf_example in enumerate(exp_g.cf_examples_list):
        if cf_example and cf_example.final_cfs_df is not None:
            cf_example.final_cfs_df['original_index'] = k.loc[k.index[i], 'Id']
            all_counterfactuals.append(cf_example.final_cfs_df)
        else:
            print(f'não foi possível gerar os contrafactuais para o exemplo: ')

if all_counterfactuals:
    concatenate_df = pd.concat(all_counterfactuals)
    concatenate_df.to_csv('C:\\Users\Cristiane\Documents\\UFMG\Arquivos_Dissertacao\Projeto\data\\novo\\counterfactuals3.csv',index=True)

exp_g = exp.generate_counterfactuals(k, total_CFs=3, desired_class=4, proximity_weight=20,  # Optional: Set proximity weight
    diversity_weight=1.0 )
all_counterfactuals = []

for i, cf_example in enumerate(exp_g.cf_examples_list):
        if cf_example and cf_example.final_cfs_df is not None:
            cf_example.final_cfs_df['original_index'] = k.loc[k.index[i], 'Id']
            all_counterfactuals.append(cf_example.final_cfs_df)
        else:
            print(f'não foi possível gerar os contrafactuais para o exemplo: ')

if all_counterfactuals:
        concatenate_df = pd.concat(all_counterfactuals)
        concatenate_df.to_csv('C:\\Users\Cristiane\Documents\\UFMG\Arquivos_Dissertacao\Projeto\data\\novo\\counterfactuals4.csv', index=True)
        '''
