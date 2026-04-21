

## Estrutura do projeto

```
nids/
├── src/
│   ├── main.py            ← ponto de entrada, orquestra tudo
│   ├── data_manager.py    ← carrega, limpa e prepara o dataset
│   ├── model_manager.py   ← treina e avalia os modelos de ML
│   ├── log_manager.py     ← salva resultados e logs
│   └── utils/
│       └── constants.py   ← configurações centrais (paths, hiperparâmetros)
├── data/                  ← CSVs do CIC-IDS2017 
├── output/                ← resumo dos experimentos com as metricas finais
├── log/                   ← resultados detalhados em JSON, para cada modelo
│   └── decision_tree/
│   └── random_forest/
│   └── mlp/
└── requirements.txt
```

## Como usar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar o dataset para o modo offline
Edite `src/utils/constants.py` e aponte `DATASET_FILE` para o seu CSV:
```python
DATASET_FILE = 'data/merged_dataset.csv'
# ou um dia específico:
DATASET_FILE = 'data/Friday-WorkingHours.pcap_ISCX.csv'
```

### 2. Configurar o dataset para o modo online
Edite `src/utils/constants.py` e aponte `ONLINE_DATASET_FILE` para o seu CSV:
```python
ONLINE_DATASET_FILE = 'data/merged_dataset.csv'
# ou um dia específico:
DATASET_FILE = 'data/Friday-WorkingHours.pcap_ISCX.csv'
```

### 3. Executar

```bash
cd src/

# Roda binário + multiclasse (padrão)
python main.py

# Só classificação binária
python main.py --mode binary

# Só multiclasse
python main.py --mode multiclass

# Especificar dataset direto na linha de comando
python main.py --dataset ../data/Friday-WorkingHours.pcap_ISCX.csv --mode both

# Especificar qual modelo (padrão todos)
python main.py --model decision_tree
```

## Modelos implementados

| Modelo | Arquivo |
|---|---|
| Decision Tree | `model_manager.py` |
| Random Forest | `model_manager.py` |
| Multilayer Perceptron | `model_manager.py` |

## Métricas avaliadas

- Acurácia
- Precisão
- Recall
- F1-Score
- Tempo de inferência
