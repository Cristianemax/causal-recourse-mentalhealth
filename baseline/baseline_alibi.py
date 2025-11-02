import torch
import pandas as pd
import numpy as np

from alibi.explainers import Counterfactual

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics import Accuracy, MaxMetric
from sklearn.preprocessing import StandardScaler

import importlib
if importlib.util.find_spec('ipywidgets') is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm

import tensorflow as tf

# Habilitar a compatibilidade com TensorFlow 1.x
tf.compat.v1.disable_eager_execution()

# Agora você pode usar get_session()
session = tf.compat.v1.keras.backend.get_session()

# Desativar a execução ansiosa
tf.compat.v1.disable_eager_execution()

# Usar placeholder
x = tf.compat.v1.placeholder(tf.float32, shape=(None, 10))

# Definir uma operação simples
y = x * 2

# Criar uma sessão para executar o gráfico
with tf.compat.v1.Session() as sess:
    result = sess.run(y, feed_dict={x: [[1]*10]})
    print(result)

dataframe = pd.read_csv('C:\\Users\Cristiane\Documents\\UFMG\Arquivos_Dissertacao\Projeto\data\\final_novo.csv', sep=';')

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
        self.embbeding_dim = input_dataframe.max().max() + 1

        y = input_dataframe[target].values
        x = input_dataframe.drop(target, axis=1).values

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
        target = torch.zeros(self.classification_dim)
        if self.split == "train":
            target[self.y_train[idx]] = 1
            return (torch.tensor(self.x_train[idx]), target)
        elif self.split == "test":
            target[self.y_test[idx]] = 1
            return (torch.tensor(self.x_test[idx]), target)
        else:
            raise ValueError("Split must be train or test")


# instanciando o dataset
train_dataset = MyDataset(df_suic, split="train", target="Suicidio", ignore_columns=[], train_ratio=0.8)
test_dataset = MyDataset(df_suic, split="test", target="Suicidio", ignore_columns=[], train_ratio=0.8)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

#######################################################################################################
## Define a MLP model with N layers: rede neural de 2 camadas

import torch.nn as nn
import torch.nn.functional as F


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


###########################################################################################################
# Define a Model with a embbeding layer and a MLP

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
       # print(f"Input to model: {x.shape}")
        x = self.embbeding_layer(x)
        print(f"After embedding: {x.shape}")

        x = x.view(x.shape[0], -1)

        x = self.mlp(x)
        ## classification
        x = F.softmax(x, dim=1)
        return x


###########################################################################################################
# test the model
example_batch = next(iter(train_loader))
example_data, example_targets = example_batch

model = ClassificationModel(train_dataset.data_dim, train_dataset.classification_dim, train_dataset.embbeding_dim,
                            hidden_out=20, hidden_dim=128, n_layers=4)  # Cadar tirou
print('model:', model, '\n')

print("Batch shape:", example_data.shape, '\n')
res = model(example_data)
print("Output shape:", res.shape, '\n')

###########################################################################################################
## Make Lightning Module
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class BaseModel(LightningModule):
    """A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Validation loop (validation_step)
        - Train loop (training_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)
    """

    def __init__(self, input_dim, output_dim, embedding_dim, embedding_out, hidden_dim):
        super().__init__()
        self.model = ClassificationModel(input_dim, output_dim, embedding_dim, embedding_out, hidden_dim=hidden_dim,
                                         n_layers=2)
        self.lr = 1e-3

        self.save_hyperparameters()

        self.accuracy = Accuracy()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # for averaging loss across batches
        # self.train_loss = MeanMetric()
        # self.val_loss = MeanMetric()

    def step(self, batch):
        x, y = batch
        y_hat = self.model(x).squeeze().float()
        # loss function
        loss = F.binary_cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y.int())
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    # gradiente
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def forward(self, x):
        """A forward pass will return the outputs of your model."""
        return self.model(x)

###########################################################################################################
# Import trainer
from pytorch_lightning.trainer import Trainer

# Initialize model
model = BaseModel(input_dim=train_dataset.data_dim, output_dim=train_dataset.classification_dim, embedding_dim=100,
                  embedding_out=64, hidden_dim=128)
print('model:', model, '\n')

###########################################################################################################
# Import callbacks
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

print("Cuda:", torch.cuda.is_available())

# Initialize callbacks

# Salve o modelo periodicamente monitorando uma quantidade.
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='best-checkpoint',
    save_top_k=1,
    mode='min',
)

# Monitore uma métrica e interrompa o treinamento quando ela parar de melhorar.
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.05,
    patience=10,
    verbose=False,
    mode='min'
)

callbacks = [checkpoint_callback, early_stopping]
# callbacks = []


# Initialize a trainer
trainer = Trainer(
    accelerator='cpu',
    devices=1,
    check_val_every_n_epoch=10,
    log_every_n_steps=10,
    callbacks=callbacks,
    auto_lr_find=True,
    enable_progress_bar=False
)

# Train the model
trainer.fit(model, train_loader, test_loader)

from sklearn.preprocessing import OneHotEncoder

# Instantiate `Counterfactual` with the modification for categorical features
# Define the prediction function
def predict_fn(X):
    model.eval()  # Colocar o modelo em modo de avaliação
    try:
        # Certifique-se de que X seja um numpy array antes de converter para tensor
        if isinstance(X, np.ndarray):
            X_tensor = torch.LongTensor(X).to(model.device)  # Convertendo X para tensor longo
        else:
            raise ValueError("Input data is not a numpy array.")

        # Verifique a forma do tensor
        print(f"X_tensor shape: {X_tensor.shape}")

        # Realizando a previsão com o modelo
        with torch.no_grad():
            prediction = model(X_tensor).cpu().numpy()

        # Verifique a forma da previsão
        print(f"Prediction shape: {prediction.shape}")

        return prediction

    except IndexError as e:
        print(f"Index error during prediction: {e}")
        # Retorna uma matriz de zeros com a forma correta no caso de erro
        return np.zeros((X.shape[0], model.model.output_dim))




filtered_df = df_suic.drop(['Suicidio'], axis=1)[:10] # Seleciona as primeiras 10 linhas
X_suic = filtered_df.values  # Converte o DataFrame filtrado em uma matriz NumPy
original_indices = filtered_df.index.tolist()  # Armazena os índices originais
target = df_suic['Suicidio'][:10]
# Defina a classe alvo para a qual você deseja gerar contrafactuais
#target_class = 0  # Exemplo: classe 0
scaler = StandardScaler()
#scaler.fit(train_dataset.x_train)
#predict_fn2 = lambda x: model(scaler.transform(x))


# List of valid target classes different from the original target (assumed to be 0)
valid_target_classes = [0]#[0, 1, 2, 3, 4] todo

all_counterfactuals = []

# Agrupar exemplos por classe original
#print(X_suic.shape())
# Agrupar exemplos por classe original
class_groups = {}

# Group examples by original target
for idx in original_indices:
    if idx < len(df_suic):  # Check against the original DataFrame, not filtered
        example = X_suic[original_indices.index(idx)].reshape(1, -1)  # Use the index from original_indices
        original_target = target.iloc[idx] # Access the original target using idx
        if original_target not in class_groups:
            class_groups[original_target] = []

        class_groups[original_target].append((idx, example))
 # Inicializar o Counterfactual para a classe atual
target_class=0
cf = Counterfactual(
                    predict_fn,
                    target_class=target_class,
                    shape=(1, X_suic.shape[1]),
                    lam_init=0.00001,
                    max_lam_steps=10,
                    learning_rate_init=0.1,
                    max_iter=1000,
                    tol=0.01,
                    early_stop=100
                )

for idx in original_indices:
    if idx < len(df_suic) and idx < len(X_suic):
        try:
            example = X_suic[idx].reshape(1, -1)
            original_target = df_suic.iloc[idx]['Suicidio']
            # Resto do código...
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
    else:
        print(f"Índice {idx} fora dos limites para os tamanhos len(df_suic)={len(df_suic)}, len(X_suic)={len(X_suic)}")

# Generate counterfactuals using the filtered data
for target_class in valid_target_classes:
    for original_target, examples in class_groups.items():
        for idx, example in examples:
            try:
                print(f"Processing example at index: {idx}")
                print(f"Example shape: {example.shape}")
                print(f"Example values: {example.flatten()}")

                # Tente gerar um contrafactual
                try:
                    print(f"Attempting to explain example at index {idx} with target class {target_class}")
                    counterfactuals = cf.explain(example)
                except Exception as e:
                    print(f"Error generating counterfactual for example {idx} with target class {target_class}: {e}")

                if counterfactuals is not None and hasattr(counterfactuals, 'cf'):
                    cf_dict = counterfactuals.cf
                    if cf_dict is not None and isinstance(cf_dict, dict):
                        original_features = example.flatten()

                        # Clipping das características geradas
                        new_features = np.clip(cf_dict['X'].flatten(), 0, 4).astype(int)

                        # Previsão do novo target
                        new_target_probabilities = predict_fn(new_features.reshape(1, -1))
                        new_target = new_target_probabilities.argmax(axis=1)[0]

                        # Armazenar apenas se new_target for diferente do original_target
                        if new_target != original_target:
                            all_counterfactuals.append({
                                'original_index': idx,
                                'feature_testada': original_features,
                                'contrafactual': new_features,
                                'original_target': original_target,
                                'new_target': new_target,
                                'distance': cf_dict['distance'],
                                'lambda': cf_dict['lambda'],
                                'index': cf_dict['index'],
                                'class': cf_dict['class'],
                                'proba': cf_dict['proba'].flatten(),
                                'loss': cf_dict['loss']
                            })
                            break  # Para após encontrar o primeiro contrafactual válido

            except Exception as e:
                print(f"Original features: {example.flatten()}")
                print(f"Error generating counterfactual for example {idx} with target class {target_class}: {e}")
                print(f"Original features: {example}")
                print(f"Predicted class: {original_target}")
# Filtrar e exportar contrafactuais válidos conforme necessário
valid_counterfactuals = []
for cf in all_counterfactuals:
    if cf is not None:
        cf['feature_testada'] = cf['feature_testada'].tolist()
        cf['contrafactual'] = cf['contrafactual'].tolist()
        valid_counterfactuals.append(cf)

# Saving valid counterfactuals to DataFrame if any are found
if valid_counterfactuals:
    counterfactuals_df = pd.DataFrame(valid_counterfactuals)
    filename = "contrafactuais_classe_variadas2.csv"
    counterfactuals_df.to_csv(filename, index=False)
    print(f"Contrafactuais exportados para '{filename}'")
else:
    print("Nenhum contrafactual válido encontrado.")
