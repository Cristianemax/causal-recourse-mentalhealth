import torch
import pandas as pd
import numpy as np
import dice_ml

from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Accuracy, MaxMetric

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

        for coll in self.ignore_columns:
            if coll in input_dataframe.columns:
                input_dataframe = input_dataframe.drop(coll, axis=1)

        self.classification_dim = len(input_dataframe[self.target].unique())
        self.data_dim = len(input_dataframe.columns) - 1
        self.embedding_dim = input_dataframe.max().max() + 1

        y = input_dataframe[target].values
        x = input_dataframe.drop(target, axis=1).values

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=1-train_ratio, random_state=42)

    def __len__(self):
        if self.split == "train":
            return len(self.x_train)
        elif self.split == "test":
            return len(self.x_test)
        else:
            raise ValueError("Split must be train or test")

    def __getitem__(self, idx):
        target = torch.zeros(self.classification_dim)
        if self.split == "train":
            target[self.y_train[idx]] = 1
            return (torch.tensor(self.x_train[idx]), target)
        elif self.split == "test":
            target[self.y_test[idx]] = 1
            return (torch.tensor(self.x_test[idx]), target)
        else:
            raise ValueError("Split must be train or test")

# Instantiate the dataset
train_dataset = MyDataset(df_suic, split="train", target="Suicidio", ignore_columns=[], train_ratio=0.8)
test_dataset = MyDataset(df_suic, split="test", target="Suicidio", ignore_columns=[], train_ratio=0.2)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # Shuffle training data
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_layers=2):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        for i in range(self.n_layers - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.Dropout(0.5))
            self.layers.append(nn.LeakyReLU())

        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim, embbeding_dim, hidden_out, hidden_dim=128, n_layers=2):
        super(ClassificationModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embbeding_dim = embbeding_dim
        self.embbeding_out = hidden_out
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embbeding_layer = nn.Embedding(self.embbeding_dim, self.embbeding_out)
        self.mlp = MLP(self.input_dim * self.embbeding_out, self.output_dim, self.hidden_dim, self.n_layers)

    def forward(self, x):
        x = self.embbeding_layer(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        ## classification
        x = F.softmax(x, dim=1)
        return x


class BaseModel(LightningModule):
    def __init__(self, input_dim, output_dim, embedding_dim, embedding_out, hidden_dim):
        super().__init__()
        self.model = ClassificationModel(input_dim, output_dim, embedding_dim, embedding_out, hidden_dim=hidden_dim, n_layers=2)
        self.lr = 1e-3
        self.save_hyperparameters()
        self.accuracy = Accuracy()  # Corrigir para o número certo de classes (0 a 4)
        self.val_acc_best = MaxMetric()

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.cross_entropy(y_hat, y.argmax(dim=1))  # Use cross entropy
        acc = self.accuracy(y_hat, y.argmax(dim=1))
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def __getitem__(self, idx):
    target = torch.zeros(self.classification_dim)
    if self.split == "train":
        target[self.y_train[idx]] = 1
        return (torch.tensor(self.x_train[idx], dtype=torch.long), target)  # Corrigido para torch.long
    elif self.split == "test":
        target[self.y_test[idx]] = 1
        return (torch.tensor(self.x_test[idx], dtype=torch.long), target)  # Corrigido para torch.long
    else:
        raise ValueError("Split must be train or test")



# Initialize model
model = BaseModel(
    input_dim=train_dataset.data_dim,
    output_dim=train_dataset.classification_dim,
    embedding_dim=100,
    embedding_out=64,
    hidden_dim=128
)
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Initialize callbacks
checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints/', filename='best-checkpoint', save_top_k=1, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.05, patience=10, mode='min')

# Initialize a trainer
trainer = Trainer(
    devices=1,
    check_val_every_n_epoch=10,
    log_every_n_steps=10,
    callbacks=[checkpoint_callback, early_stopping],
    enable_progress_bar=False
)

# Train the model
trainer.fit(model, train_loader, test_loader)
torch.save(model, "model.pt")

# Prepare for DICE
d = dice_ml.Data(dataframe=df_suic, continuous_features=[
    "Suic_familia", "Capaz de tomar decisões importantes", "Estudante",
    "Hipocondriase", "Sentimentos_culpa", "Trabalho e interesses",
    'Dep_familia', 'Alc_familia', 'Capaz de desfrutar das coisas', 'Droga'
], outcome_name='Suicidio')

m = dice_ml.Model(model=model, backend='PYT')
exp = dice_ml.Dice(d, m)

# Create a sample for counterfactual generation
k = df_suic.drop(columns='Suicidio')
k = k.astype(int)

class MyDataset(Dataset):

    def __init__(self, input_dataframe, split="train", target="Suicidio", ignore_columns=[], train_ratio=0.8):
        self.split = split
        self.target = target
        self.ignore_columns = ignore_columns

        for coll in self.ignore_columns:
            if coll in input_dataframe.columns:
                input_dataframe = input_dataframe.drop(coll, axis=1)

        self.classification_dim = len(input_dataframe[self.target].unique())
        self.data_dim = len(input_dataframe.columns) - 1
        self.embedding_dim = input_dataframe.max().max() + 1

        y = input_dataframe[target].values
        x = input_dataframe.drop(target, axis=1).values

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=1-train_ratio, random_state=42)

    def __len__(self):
        return len(self.x_train) if self.split == "train" else len(self.x_test)

    def __getitem__(self, idx):
        target = torch.zeros(self.classification_dim)
        if self.split == "train":
            target[self.y_train[idx]] = 1
            return (torch.tensor(self.x_train[idx], dtype=torch.long), target)  # Aqui está a correção
        elif self.split == "test":
            target[self.y_test[idx]] = 1
            return (torch.tensor(self.x_test[idx], dtype=torch.long), target)  # Correção também aqui
        else:
            raise ValueError("Split must be train or test")

exp_g = exp.generate_counterfactuals(k, total_CFs=10, desired_class=0)
all_counterfactuals = []

for i, cf_example in enumerate(exp_g.cf_examples_list):
        if cf_example and cf_example.final_cfs_df is not None:
            cf_example.final_cfs_df['original_index'] = df_suic.index[i]
            all_counterfactuals.append(cf_example.final_cfs_df)
        else:
            print(f'não foi possível gerar os contrafactuais para o exemplo: ')#{len(i)}')

if all_counterfactuals:
    concatenate_df = pd.concat(all_counterfactuals)
    concatenate_df.to_csv('/content/counterfactuals0.csv',index=True)


exp_g = exp.generate_counterfactuals(k, total_CFs=10, desired_class=1)

print(f'verifica tamanho da lista de exemplos contrafactuais: {len(exp_g.cf_examples_list)}')

all_counterfactuals = []

for i, cf_example in enumerate(exp_g.cf_examples_list):
        if cf_example and cf_example.final_cfs_df is not None:
            cf_example.final_cfs_df['original_index'] = df_suic.index[i]
            all_counterfactuals.append(cf_example.final_cfs_df)
        else:
            print(f'não foi possível gerar os contrafactuais para o exemplo: ')#{len(i)}')

if all_counterfactuals:
    concatenate_df = pd.concat(all_counterfactuals)
    concatenate_df.to_csv('/content/counterfactuals1.csv',index=True)


exp_g = exp.generate_counterfactuals(k, total_CFs=10, desired_class=2)

print(f'verifica tamanho da lista de exemplos contrafactuais: {len(exp_g.cf_examples_list)}')

all_counterfactuals = []

for i, cf_example in enumerate(exp_g.cf_examples_list):
        if cf_example and cf_example.final_cfs_df is not None:
            cf_example.final_cfs_df['original_index'] = df_suic.index[i]
            all_counterfactuals.append(cf_example.final_cfs_df)
        else:
            print(f'não foi possível gerar os contrafactuais para o exemplo: ')#{len(i)}')

if all_counterfactuals:
    concatenate_df = pd.concat(all_counterfactuals)
    concatenate_df.to_csv('/content/counterfactuals2.csv',index=True)

exp_g = exp.generate_counterfactuals(k, total_CFs=10, desired_class=3)

print(f'verifica tamanho da lista de exemplos contrafactuais: {len(exp_g.cf_examples_list)}')

all_counterfactuals = []

for i, cf_example in enumerate(exp_g.cf_examples_list):
        if cf_example and cf_example.final_cfs_df is not None:
            cf_example.final_cfs_df['original_index'] = df_suic.index[i]
            all_counterfactuals.append(cf_example.final_cfs_df)
        else:
            print(f'não foi possível gerar os contrafactuais para o exemplo: ')#{len(i)}')

if all_counterfactuals:
    concatenate_df = pd.concat(all_counterfactuals)
    concatenate_df.to_csv('/content/counterfactuals3.csv',index=True)

    exp_g = exp.generate_counterfactuals(k, total_CFs=10, desired_class=4)

    print(f'verifica tamanho da lista de exemplos contrafactuais: {len(exp_g.cf_examples_list)}')

    all_counterfactuals = []

    for i, cf_example in enumerate(exp_g.cf_examples_list):
        if cf_example and cf_example.final_cfs_df is not None:
            cf_example.final_cfs_df['original_index'] = df_suic.index[i]
            all_counterfactuals.append(cf_example.final_cfs_df)
        else:
            print(f'não foi possível gerar os contrafactuais para o exemplo: ')  # {len(i)}')

    if all_counterfactuals:
        concatenate_df = pd.concat(all_counterfactuals)
        concatenate_df.to_csv('/content/counterfactuals4.csv', index=True)

