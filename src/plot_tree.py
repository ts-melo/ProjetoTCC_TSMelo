"""
plot_tree.py
------------
Carrega o modelo Decision Tree salvo e gera uma imagem da árvore
com profundidade limitada para visualização dos nós de decisão.

Uso:
    cd src
    python plot_tree.py --mode binary
    python plot_tree.py --mode multiclass --depth 5
    python plot_tree.py --mode binary --depth 3
"""

import argparse
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

import utils.constants as CONSTANTS
from data_manager import DataManager


def main():
    parser = argparse.ArgumentParser(description='Visualiza a Decision Tree treinada')
    parser.add_argument('--mode',  type=str, default='binary',
                        choices=['binary', 'multiclass'],
                        help='Modo de classificação do modelo a carregar')
    parser.add_argument('--depth', type=int, default=CONSTANTS.DT_PLOT_DEPTH,
                        help='Profundidade máxima a exibir (padrão: DT_PLOT_DEPTH do constants)')
    args = parser.parse_args()

    # ── 1. Carrega o modelo salvo ─────────────────────────────────────────────
    model_path = CONSTANTS.MODELS_FOLDER + f"decision_tree_{args.mode}.pkl"
    if not os.path.exists(model_path):
        print(f"[plot_tree] Modelo não encontrado em: {model_path}")
        print(f"[plot_tree] Treine e salve o modelo primeiro rodando main.py")
        return

    print(f"[plot_tree] Carregando modelo: {model_path}")
    model = joblib.load(model_path)

    # ── 2. Carrega os nomes das features e classes ────────────────────────────
    print(f"[plot_tree] Carregando nomes de features do dataset...")
    data = DataManager()
    data.load()
    data.clean()
    data.prepare(mode=args.mode)

    feature_names = data.features
    if args.mode == 'binary':
        class_names = ['BENIGN', 'ATTACK']
    else:
        class_names = list(data.label_encoder.classes_)

    print(f"[plot_tree] Features: {len(feature_names)}")
    print(f"[plot_tree] Classes: {class_names}")

    # ── 3. Gera a imagem ──────────────────────────────────────────────────────
    # A árvore completa tem milhares de nós — limitamos a max_depth níveis
    # para que os nós de decisão principais sejam legíveis.
    # max_depth=4 mostra as 4 primeiras divisões que explicam a maior parte
    # da capacidade discriminativa do modelo.

    fig_width  = max(20, 2 ** args.depth * 3)
    fig_height = args.depth * 4
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    plot_tree(
        model,
        max_depth=args.depth,           # limita profundidade para legibilidade
        feature_names=feature_names,    # nomes das 78 features
        class_names=class_names,        # nomes das classes
        filled=True,                    # colore os nós pela classe majoritária
        rounded=True,                   # bordas arredondadas
        fontsize=10,
        ax=ax,
        impurity=True,                  # mostra o índice Gini de cada nó
        proportion=False,               # mostra contagem absoluta, não proporção
    )

    ax.set_title(
        f"Decision Tree — modo {args.mode} — primeiros {args.depth} níveis\n"
        f"(árvore completa tem profundidade {model.get_depth()} e "
        f"{model.get_n_leaves()} folhas)",
        fontsize=13
    )

    plt.tight_layout()
    os.makedirs(CONSTANTS.OUTPUT_FOLDER, exist_ok=True)
    path = CONSTANTS.OUTPUT_FOLDER + f"decision_tree_{args.mode}_depth{args.depth}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[plot_tree] Imagem salva → {path}")
    print(f"[plot_tree] Árvore completa: profundidade={model.get_depth()}, folhas={model.get_n_leaves():,}")
    print(f"[plot_tree] Visualizando: {args.depth} níveis dos {model.get_depth()} totais")


if __name__ == '__main__':
    main()