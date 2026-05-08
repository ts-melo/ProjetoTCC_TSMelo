"""Uso:
    cd src
    python plot_lambda.py
    python plot_lambda.py --mode binary --lambdas 5 10 20 50 --steps 1000
    python plot_lambda.py --model decision_tree
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import os
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import utils.constants as CONSTANTS
from data_manager  import DataManager
from model_manager import ModelManager
from task_manager  import TaskManager

COLORS = {
    'decision_tree': '#2E5FA3',
    'random_forest': '#1D9E75',
    'mlp':           '#D85A30',
}
LABELS = {
    'decision_tree': 'Decision Tree',
    'random_forest': 'Random Forest',
    'mlp':           'MLP',
}


def sim_lambda(models_dict, batches, label_names):
   
    results = {}

    for classifier_name, classifier in models_dict.items():
        step_log   = []
        sim_start  = time.time()
        cum_correct = 0
        cum_total   = 0

        for step, arrived, batch in batches:
            if not batch:
                continue

            X_batch     = np.array([f['features'] for f in batch])
            true_labels = np.array([f['true_label'] for f in batch])
            predictions = classifier.predict(X_batch)

            cum_correct += int((predictions == true_labels).sum())
            cum_total   += len(batch)

            step_log.append({
                'step':         step,
                'cum_accuracy': cum_correct / cum_total * 100 if cum_total else 0,
                'cum_cpu_time': time.time() - sim_start,
            })

        results[classifier_name] = step_log

    return results


def collect(models_dict, X_online, y_online, label_names, lambdas, n_steps, seed):
   
    all_results = {}

    for lam in lambdas:
        print(f"\n[plot_lambda] λ = {lam}")

        tasks = TaskManager(rate=lam, seed=seed)
        tasks.load_flows(X_online, y_online, feature_names=[])
        tasks.generate_arrivals(n_steps)

        batches = []
        for step in range(n_steps):
            arrived = tasks.step()
            batch   = tasks.drain_pending()
            batches.append((step, arrived, batch))

        total = sum(len(b) for _, _, b in batches)
        print(f"  {total:,} fluxos em {n_steps} steps")

        all_results[lam] = sim_lambda(models_dict, batches, label_names)

        for name, log in all_results[lam].items():
            final_acc  = log[-1]['cum_accuracy'] if log else 0
            final_time = log[-1]['cum_cpu_time']  if log else 0
            print(f"  {name:<22} acc={final_acc:.3f}%  time={final_time:.3f}s")

    return all_results


def downsample(log, n_points=50):
    if len(log) <= n_points:
        return log
    idx = np.linspace(0, len(log) - 1, n_points, dtype=int)
    return [log[i] for i in idx]



def plot_acc_bars(all_results, mode, output_folder):
    """
    Barras agrupadas por λ. Eixo Y começa em 99% para destacar diferenças.
    Cada grupo tem uma barra por modelo.

    Por que eixo Y em 99%?
    Com acurácias entre 99.6% e 99.9%, um eixo começando em 0%
    faz todas as barras parecerem idênticas. Começar em 99% amplia
    a região de interesse e torna as diferenças visíveis.
    """
    lambdas   = sorted(all_results.keys())
    models    = list(next(iter(all_results.values())).keys())
    n_models  = len(models)
    x         = np.arange(len(lambdas))
    width     = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model_name in enumerate(models):
        accs = []
        for lam in lambdas:
            log = all_results[lam][model_name]
            accs.append(log[-1]['cum_accuracy'] if log else 0)

        bars = ax.bar(
            x + i * width - (n_models - 1) * width / 2,
            accs,
            width,
            label=LABELS.get(model_name, model_name),
            color=COLORS.get(model_name, 'gray'),
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5,
        )

        # Valor no topo de cada barra
        for bar, val in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=8, color='#333333'
            )

    ax.set_xlabel('Taxa de chegada λ (fluxos/step)', fontsize=12)
    ax.set_ylabel('Acurácia (%)', fontsize=12)
    ax.set_title(f'Acurácia final por λ — modo {mode}\n'
                 f'(eixo Y a partir de 99% para destacar diferenças)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'λ={l}' for l in lambdas])

    # Eixo Y quebrado em 99% — amplia a região de diferença
    ax.set_ylim(99.0, 100.2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = output_folder + f'acc_bars_lambda_{mode}.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot_lambda] Salvo → {path}")


def plot_acc_lines(all_results, mode, output_folder):

    lambdas  = sorted(all_results.keys())
    models   = list(next(iter(all_results.values())).keys())
    n_lambda = len(lambdas)

    fig, axes = plt.subplots(1, n_lambda, figsize=(5 * n_lambda, 5), sharey=True)
    if n_lambda == 1:
        axes = [axes]

    for ax, lam in zip(axes, lambdas):
        for model_name in models:
            log      = downsample(all_results[lam][model_name])
            steps    = [e['step'] for e in log]
            accs     = [e['cum_accuracy'] for e in log]

            ax.plot(steps, accs,
                    label=LABELS.get(model_name, model_name),
                    color=COLORS.get(model_name, 'gray'),
                    linewidth=2)

        ax.set_title(f'λ = {lam}', fontsize=11)
        ax.set_xlabel('Step', fontsize=10)
        ax.set_ylim(99.0, 100.1)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Acurácia cumulativa (%)', fontsize=11)
    fig.suptitle(f'Evolução da acurácia por step — modo {mode}\n'
                 f'(eixo Y 99–100% para destacar diferenças)', fontsize=12)

    plt.tight_layout()
    path = output_folder + f'acc_lines_lambda_{mode}.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot_lambda] Salvo → {path}")


def plot_cpu_bars(all_results, mode, output_folder):
    
    lambdas  = sorted(all_results.keys())
    models   = list(next(iter(all_results.values())).keys())
    n_models = len(models)
    x        = np.arange(len(lambdas))
    width    = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model_name in enumerate(models):
        times = []
        for lam in lambdas:
            log = all_results[lam][model_name]
            times.append(log[-1]['cum_cpu_time'] if log else 0)

        bars = ax.bar(
            x + i * width - (n_models - 1) * width / 2,
            times,
            width,
            label=LABELS.get(model_name, model_name),
            color=COLORS.get(model_name, 'gray'),
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5,
        )

        for bar, val in zip(bars, times):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.2f}s',
                ha='center', va='bottom', fontsize=8, color='#333333'
            )

    ax.set_xlabel('Taxa de chegada λ (fluxos/step)', fontsize=12)
    ax.set_ylabel('Tempo total de simulação (s)', fontsize=12)
    ax.set_title(f'Tempo de simulação por λ — modo {mode}', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'λ={l}' for l in lambdas])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = output_folder + f'cpu_bars_lambda_{mode}.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot_lambda] Salvo → {path}")


def plot_cpu_lines(all_results, mode, output_folder):
    
    lambdas  = sorted(all_results.keys())
    models   = list(next(iter(all_results.values())).keys())
    n_lambda = len(lambdas)

    fig, axes = plt.subplots(1, n_lambda, figsize=(5 * n_lambda, 5), sharey=False)
    if n_lambda == 1:
        axes = [axes]

    for ax, lam in zip(axes, lambdas):
        for model_name in models:
            log   = downsample(all_results[lam][model_name])
            steps = [e['step'] for e in log]
            times = [e['cum_cpu_time'] for e in log]

            ax.plot(steps, times,
                    label=LABELS.get(model_name, model_name),
                    color=COLORS.get(model_name, 'gray'),
                    linewidth=2)

        ax.set_title(f'λ = {lam}', fontsize=11)
        ax.set_xlabel('Step', fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Tempo CPU acumulado (s)', fontsize=11)
    fig.suptitle(f'Evolução do tempo CPU por step — modo {mode}', fontsize=12)

    plt.tight_layout()
    path = output_folder + f'cpu_lines_lambda_{mode}.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot_lambda] Salvo → {path}")


def main():
    parser = argparse.ArgumentParser(description='Gera gráficos de acurácia e tempo por λ')
    parser.add_argument('--mode',    type=str, default='binary',
                        choices=['binary', 'multiclass'])
    parser.add_argument('--steps',   type=int, default=CONSTANTS.SIM_STEPS,
                        help='Número de steps por simulação')
    parser.add_argument('--seed',    type=int, default=CONSTANTS.RANDOM_STATE)
    parser.add_argument('--lambdas', type=float, nargs='+',
                        default=[5, 10, 15],
                        help='Valores de λ (ex: --lambdas 5 10 20 50)')
    parser.add_argument('--model',   type=str, default=None,
                        choices=['decision_tree', 'random_forest', 'mlp'],
                        help='Roda só um modelo. Omitir = todos.')
    args = parser.parse_args()

    lambdas = sorted(args.lambdas)
    print(f"\n{'='*60}")
    print(f"  PLOT LAMBDA — modo={args.mode}  steps={args.steps}")
    print(f"  λ values: {lambdas}")
    print(f"{'='*60}")

    # 1. Carrega e limpa os dados
    data = DataManager()
    data.load()
    data.clean()

    # 2. Treina os modelos UMA VEZ
    print(f"\n[plot_lambda] Treinando modelos...")
    data.prepare(mode=args.mode)
    X_train, X_test, y_train, y_test = data.get_split()
    label_names = data.label_names() if args.mode == 'multiclass' else ['BENIGN', 'ATTACK']

    mm = ModelManager()
    mm.train_all(X_train, y_train, only=args.model)
    mm.evaluate_all(X_test, y_test, mode=args.mode, label_names=label_names)
    mm.compare()

    models_dict = mm.models

    # 3. Prepara dados online
    if CONSTANTS.ONLINE_DATASET_FILE:
        print(f"\n[plot_lambda] Carregando dataset online...")
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

    # 4. Coleta resultados para cada λ
    all_results = collect(
        models_dict, X_online, y_online,
        label_names, lambdas, args.steps, args.seed
    )

    # 5. Gera os quatro gráficos
    os.makedirs(CONSTANTS.OUTPUT_FOLDER, exist_ok=True)
    plot_acc_bars(all_results,  args.mode, CONSTANTS.OUTPUT_FOLDER)
    plot_acc_lines(all_results, args.mode, CONSTANTS.OUTPUT_FOLDER)
    plot_cpu_bars(all_results,  args.mode, CONSTANTS.OUTPUT_FOLDER)
    plot_cpu_lines(all_results, args.mode, CONSTANTS.OUTPUT_FOLDER)

    print(f"\n✓ 4 gráficos salvos em {CONSTANTS.OUTPUT_FOLDER}\n")


if __name__ == '__main__':
    main()