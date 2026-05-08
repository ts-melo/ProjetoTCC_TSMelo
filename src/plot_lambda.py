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
        attack_counts = defaultdict(int)
        step_log = []
        sim_start_time = time.time()

        for step, arrived, batch in batches:
            if not batch:
                continue
            
            X_batch = np.array([item['features'] for item in batch])
            true_labels = np.array([item['true_label'] for item in batch])
            predictions = classifier.predict(X_batch)

            n_attacks = 0

            for pred in predictions:
                pred_int = int(pred)
                if pred_int != 0:
                    n_attacks += 1
                    label_str = label_names[pred_int]
                    attack_counts[label_str] += 1

            n_correct = int((predictions == true_labels).sum())
            step_log.append({
                'classified': len(batch),
                'correct' : n_correct,
            })

            sim_time = round(time.time() - sim_start_time, 4)
            total_classified = sum(s['classified'] for s in step_log)
            total_correct = sum(s['correct'] for s in step_log)
            accuracy = round(total_correct/total_classified * 100, 4) if total_classified else 0
            results[classifier_name] = {
                'accuracy' : accuracy,
                'time_s' : sim_time
            }
    return results

def collect(models_dict, X_online, y_online,label_names, lambdas, n_steps, seed=42):
    data_acc = {name:[] for name in models_dict}
    data_time = {name:[] for name in models_dict}

    for lam in lambdas:
        print(f"\n[plot_lambda] lambda={lam} ({'-'*40})")

        tasks = TaskManager(rate=lam, seed=seed)
        tasks.load_flows(X_online, y_online, feature_names=[])
        tasks.generate_arrivals(n_steps)

        batches = []
        for step in range(n_steps):
            arrived = tasks.step()
            batch = tasks.drain_pending()
            batches.append((step, arrived, batch))

        total = sum(len(b) for _, _, b in batches)
        print(f"\n[plot_lambda] {n_steps} steps - Total flows generated: {total:,}")

        res = sim_lambda(models_dict, batches, label_names)

        for name, r in res.items():
            data_acc[name].append(r['accuracy'])
            data_time[name].append(r['time_s'])
            print(f"  {name:<22} acc={r['accuracy']:.2f}%  time={r['time_s']:.3f}s")

    return data_acc, data_time

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

def plot_acc(lambdas, data_acc, mode, output_folder):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, accs, in data_acc.items():

        ax.plot(lambdas, accs, label=name.replace('_', ' ').title(), marker=MARKERS.get(name, 'o'), color=COLORS.get(name, None), linewidth=2, markersize=7)
        ax.set_xlabel('taxa de chegada (λ)', fontsize=12)
        ax.set_ylabel('Acurácia (%)', fontsize=12)
        ax.set_title(f'Acurácia vs Taxa de Chegada ({mode})', fontsize=14)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(lambdas)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        path = output_folder + f'acc_lambda_{mode}.png'
        plt.savefig(path)
        plt.close()
        print(f"\n[plot_lambda] Acurácia plot saved to: {path}")

def plot_time(lambdas, data_time, mode, output_folder):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, times, in data_time.items():

        ax.plot(lambdas, times, label=name.replace('_', ' ').title(), marker=MARKERS.get(name, 'o'), color=COLORS.get(name, 'gray'), linewidth=2, markersize=7)
        ax.set_xlabel('taxa de chegada (λ)', fontsize=12)
        ax.set_ylabel('tempo de simulação (s)', fontsize=12)
        ax.set_title(f'tempo de simulação por λ - modo ({mode})', fontsize=14)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(lambdas)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        path = output_folder + f'time_by_lambda_{mode}.png'
        plt.savefig(path)
        plt.close()
        print(f"\n[plot_lambda] cputime plot saved to: {path}")

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
    data_acc, data_time = collect(
        models_dict, X_online, y_online,
        label_names, lambdas, args.steps, args.seed
    )
 
    # ── 5. Gera os gráficos ───────────────────────────────────────────────────
    os.makedirs(CONSTANTS.OUTPUT_FOLDER, exist_ok=True)
    plot_acc(lambdas, data_acc,  args.mode, CONSTANTS.OUTPUT_FOLDER)
    plot_time(lambdas, data_time, args.mode, CONSTANTS.OUTPUT_FOLDER)
 
    print(f"\n✓ Gráficos salvos em {CONSTANTS.OUTPUT_FOLDER}\n")
 
 
if __name__ == '__main__':
    main()
 