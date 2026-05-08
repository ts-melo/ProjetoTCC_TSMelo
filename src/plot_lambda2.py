import sys
import os
import time
from collections import defaultdict
import argparse
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

import utils.constants as CONSTANTS
from data_manager import DataManager
from model_manager import ModelManager
from task_manager import TaskManager

def sim_lambda(models_dict, batches, label_names):
    results = {}

    for classifier_name, classifier in models_dict.items():
        step_log = []
        sim_start_time = time.time()

        cum_correct = 0
        cum_total = 0

        for step, arrived, batch in batches:
            if not batch:
                continue
            
            X_batch = np.array([item['features'] for item in batch])
            true_labels = np.array([item['true_label'] for item in batch])
            predictions = classifier.predict(X_batch)

            n_correct = int((predictions == true_labels).sum())

            cum_correct += n_correct
            cum_total += len(batch)

            acc = cum_correct / cum_total if cum_total else 0
            cpu_time = time.time() - sim_start_time

            step_log.append({
                'step': step,
                'accuracy': acc,
                'cpu_time': cpu_time
            })

        results[classifier_name] = step_log

    return results

def collect(models_dict, X_online, y_online, label_names, lambdas, n_steps, seed=42):
    all_results = {}

    for lam in lambdas:
        print(f"\n[plot_lambda] lambda={lam}")

        tasks = TaskManager(rate=lam, seed=seed)
        tasks.load_flows(X_online, y_online, feature_names=[])
        tasks.generate_arrivals(n_steps)

        batches = []
        for step in range(n_steps):
            arrived = tasks.step()
            batch = tasks.drain_pending()
            batches.append((step, arrived, batch))

        results = sim_lambda(models_dict, batches, label_names)
        all_results[lam] = results

    return all_results

COLORS = {
    'decision_tree': 'blue',
    'random_forest': 'orange',
    'mlp': 'green'
}

MARKERS = {
    'decision_tree': 'o',
    'random_forest': 's',
    'mlp': '^'
}

def plot_acc_time(all_results):
    for lam, models in all_results.items():

        steps = [s['step'] for s in next(iter(models.values()))]
        steps = steps[::200]  # reduz quantidade

        x = np.arange(len(steps))
        width = 0.25

        plt.figure()

        for i, (model_name, data) in enumerate(models.items()):
            acc = [data[s]['accuracy'] for s in steps]
            plt.bar(x + i*width, acc, width, label=model_name)

        plt.xticks(x + width, steps)
        plt.xlabel('Tempo (steps)')
        plt.ylabel('Acurácia')
        plt.title(f'Acurácia ao longo do tempo (λ={lam})')
        plt.legend()
        plt.grid(axis='y')
        path = f"{CONSTANTS.OUTPUT_FOLDER}/acc_lambda_{lam}.png"
        plt.savefig(path, dpi=150)
        print(f"Salvo em: {path}")
        plt.show()

def plot_cpu_time(all_results):
    for lam, models in all_results.items():

        steps = [s['step'] for s in next(iter(models.values()))]
        steps = steps[::200]

        x = np.arange(len(steps))
        width = 0.25

        plt.figure()

        for i, (model_name, data) in enumerate(models.items()):
            cpu = [data[s]['cpu_time'] for s in steps]
            plt.bar(x + i*width, cpu, width, label=model_name)
       
        plt.xticks(x + width, steps)
        plt.xlabel('Tempo (steps)')
        plt.ylabel('Tempo CPU (s)')
        plt.title(f'Tempo CPU ao longo do tempo (λ={lam})')
        plt.legend()
        plt.grid(axis='y')
        path = f"{CONSTANTS.OUTPUT_FOLDER}/cpu_lambda_{lam}.png"
        plt.savefig(path, dpi=150)
        print(f"Salvo em: {path}")
        plt.show()

# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(description='Gera gráficos de acurácia e tempo por λ')
    parser.add_argument('--mode',    type=str, default='binary',
                        choices=['binary', 'multiclass'])
    parser.add_argument('--steps',   type=int, default=CONSTANTS.SIM_STEPS,
                        help='Número de steps por simulação')
    parser.add_argument('--seed',    type=int, default=CONSTANTS.RANDOM_STATE)
    parser.add_argument('--lambdas', type=float, nargs='+',
                        default=[5, 10, 15],
                        help='Valores de λ a testar (ex: --lambdas 5 10 20 50)')
    parser.add_argument('--model',   type=str, default=None,
                        choices=['decision_tree', 'random_forest', 'mlp'],
                        help='Roda só um modelo. Omitir = todos.')
    args = parser.parse_args()
 
    lambdas = sorted(args.lambdas)
    print(f"\n{'='*60}")
    print(f"  PLOT LAMBDA — modo={args.mode}  steps={args.steps}")
    print(f"  λ values: {lambdas}")
    print(f"{'='*60}")
 
    data = DataManager()
    data.load()
    data.clean()
 
    data.prepare(mode=args.mode)
    X_train, X_test, y_train, y_test = data.get_split()
    label_names = data.label_names() if args.mode == 'multiclass' else ['BENIGN', 'ATTACK']
 
    mm = ModelManager()
    mm.train_all(X_train, y_train, only=args.model)
    mm.evaluate_all(X_test, y_test, mode=args.mode, label_names=label_names)
    mm.compare()
 
    models_dict = mm.models
 
    # ── 3. Prepara dados online ───────────────────────────────────────────────
    if CONSTANTS.ONLINE_DATASET_FILE:
        print(f"\n[plot_lambda] Carregando dataset online: {CONSTANTS.ONLINE_DATASET_FILE}")
        online_data = DataManager()
        online_data.load(CONSTANTS.ONLINE_DATASET_FILE)
        online_data.clean()
 
        label_col = 'Label'
        features  = [c for c in online_data.df.columns if c != label_col]
        X_all     = online_data.df[features].select_dtypes(include=['number'])
        X_online  = data.scaler.transform(X_all)
 
        if args.mode == 'binary':
            y_online = online_data.df[label_col].apply(
                lambda x: 0 if x == CONSTANTS.BENIGN_LABEL else 1
            ).values
        else:
            y_online = data.label_encoder.transform(online_data.df[label_col])
    else:
        print(f"\n[plot_lambda] Usando test split como dados online.")
        X_online, y_online = X_test, y_test
 
    # ── 4. Coleta resultados para cada λ ──────────────────────────────────────
    all_results = collect(
        models_dict, X_online, y_online,
        label_names, lambdas, args.steps, args.seed
    )
 
    # ── 5. Gera os gráficos ───────────────────────────────────────────────────
    os.makedirs(CONSTANTS.OUTPUT_FOLDER, exist_ok=True)
    plot_acc_time(all_results)
    plot_cpu_time(all_results)
 
    print(f"\nGráficos salvos em {CONSTANTS.OUTPUT_FOLDER}\n")
 
 
if __name__ == '__main__':
    main()
 