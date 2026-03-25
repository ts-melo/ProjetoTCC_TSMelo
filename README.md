TCC — UNIFESP ICT  
Candidata: Thaís Souza de Melo  
Orientador: Joahannes B. D. da Costa

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
├── data/                  ← coloque os CSVs do CIC-IDS2017 aqui
├── output/                ← resumo dos experimentos
├── log/                   ← resultados detalhados em JSON
└── requirements.txt
```

## Como usar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar o dataset
Edite `src/utils/constants.py` e aponte `DATASET_FILE` para o seu CSV:
```python
DATASET_FILE = 'data/merged_dataset.csv'
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
```

## Modelos implementados

| Modelo | Arquivo |
|---|---|
| Decision Tree | `model_manager.py` |
| Random Forest | `model_manager.py` |

## Métricas avaliadas

- Acurácia balanceada
- Precisão
- Recall (sensibilidade)
- F1-Score
- Tempo de inferência