import argparse
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
import numpy as np
import utils.constants as CONSTANTS
from data_manager  import DataManager
from model_manager import ModelManager
from task_manager  import TaskManager
import log_manager

#training the models
def run_offline(data, current_mode): 
    print(f"\n{'='*60}")
    print(f"  OFFLINE TRAINING — {current_mode.upper()}")
    print(f"{'='*60}")

    data.prepare(mode=current_mode)
    X_train, X_test, y_train, y_test = data.get_split()
    label_names = data.label_names() if current_mode == 'multiclass' else ['BENIGN', 'ATTACK']

    models = ModelManager()
    models.train_all(X_train, y_train)
    models.evaluate_all(X_test, y_test, mode=current_mode, label_names=label_names)
    models.compare()

    return models, X_test, y_test, label_names

def run_online(models, X_online, y_online, rate, n_steps, seed, current_mode, label_names):

    print(f"\n{'='*60}")
    print(f"  ONLINE SIMULATION — {current_mode.upper()}")
    print(f"  lambda={rate} flows/step  |  {n_steps} steps")
    print(f"{'='*60}")

    classifier_name = list(models.models.keys())[0]
    classifier = models.models[classifier_name] #pega o modelo treinado da fase offline para classificar os ataques e benignos
    print(f"[Main] Classifier for online phase: {classifier_name}")

    tasks = TaskManager(rate=rate, seed=seed)
    tasks.load_flows(X_online, y_online, feature_names=[])
    tasks.generate_arrivals(n_steps)

    attack_counts = defaultdict(int)
    step_log = []

    for step in range(n_steps):
        arrived = tasks.step()
        batch = tasks.drain_pending()
        if not batch:
            continue

        X_batch     = np.array([f['features'] for f in batch]) # + ou - 10 fluxos por step é um lote
        flow_ids    = [f['flow_id'] for f in batch]
        true_labels = np.array([f['true_label'] for f in batch])

        predictions = classifier.predict(X_batch) ## classifica ataques e benignos

        tasks.record_batch(flow_ids, predictions, step)

        n_attacks = 0
        step_attack_types = defaultdict(int)
        for pred in predictions: #oq detectou
            pred_int = int(pred) #converte para inteiro (0 ou 1 no caso binário, ou 0 a n-1 no caso multiclass)
            if pred_int != 0: # 0 é benigno, então só vai escrever quando detectar um ataque
                n_attacks += 1
                if pred_int < len(label_names):
                    label_str = label_names[pred_int] #converte pro nome do ataque
                else:
                    label_str = f"class_{pred_int}"
                attack_counts[label_str] += 1
                step_attack_types[label_str] += 1
 
        n_correct = int((predictions == true_labels).sum())
        step_log.append({
            'step':             step,
            'arrived':          len(arrived),
            'classified':       len(batch),
            'attacks_detected': n_attacks,
            'correct':          n_correct,
            'attack_types':     dict(step_attack_types),
        })

        if step % max(1, n_steps // 10) == 0:
            print(f"  step {step:4d} | arrived: {len(arrived):4d} | "
                  f"classified: {len(batch):4d} | "
                  f"attacks: {n_attacks:4d} | "
                  f"correct: {n_correct:4d}")

    tasks.print_stats()

    total_classified = sum(s['classified'] for s in step_log)
    total_attacks    = sum(s['attacks_detected'] for s in step_log)
    total_correct    = sum(s['correct'] for s in step_log)
    accuracy = round(total_correct / total_classified, 4) if total_classified else 0

    print(f"\n-- Online Simulation Summary ---------------------")
    print(f"  Total flows classified : {total_classified:,}")
    print(f"  Attacks detected       : {total_attacks:,}")
    print(f"  Overall accuracy       : {accuracy:.4f}")
    print(f"--------------------------------------------------\n")

    if attack_counts:
        print(f"\n  Attack types detected:")
        for label, count in sorted(attack_counts.items(), key=lambda x: -x[1]):
            pct = count / total_attacks * 100 if total_attacks else 0
            print(f"    {label:<35} {count:>6,}  ({pct:5.1f}%)")
    else:
        print(f"\n  No attacks detected in this simulation.")
 
    print(f"--------------------------------------------------\n")
 
    online_result = {
        'total_classified': total_classified,
        'total_attacks':    total_attacks,
        'overall_accuracy': accuracy,
        'attack_type_counts': dict(attack_counts),
        'steps': step_log,
    }
 
    return online_result


def run(dataset_path=None, online_dataset_path=None, mode=None, rate=None, n_steps=None, seed=None):

    mode    = mode    or CONSTANTS.CLASSIFICATION_MODE
    rate    = rate    or CONSTANTS.POISSON_RATE
    n_steps = n_steps or CONSTANTS.SIM_STEPS
    seed    = seed    or CONSTANTS.RANDOM_STATE
    online_dataset_path = online_dataset_path or CONSTANTS.ONLINE_DATASET_FILE
    modes   = ['binary', 'multiclass'] if mode == 'both' else [mode]

    data = DataManager()
    data.load(dataset_path)
    data.clean()
    data.summary()
    if online_dataset_path:
        print(f"\n[Main] Online simulation dataset: {online_dataset_path}")
        online_data = DataManager()
        online_data.load(online_dataset_path)
        online_data.clean()
        online_data.summary()
    else:
        print(f"\n[Main] No separate online dataset provided. Using same data for simulation.")
        online_data = None

    for current_mode in modes:

        models, X_test, y_test, label_names = run_offline(data, current_mode)
        log_manager.log_results(models.get_results(), current_mode)
        log_manager.log_summary(models.get_results(), current_mode)

        if online_data is not None:
            print(f"\n[Main] Preparing online dataset with training scaler...")
            label_col = 'Label'
            features  = [c for c in online_data.df.columns if c != label_col]
            X_all     = online_data.df[features].select_dtypes(include=['number'])
            X_online  = data.scaler.transform(X_all)
            if current_mode == 'binary':
                y_online = online_data.df[label_col].apply(lambda x: 0 if x == CONSTANTS.BENIGN_LABEL else 1).values
            else:
                y_online = data.label_encoder.transform(online_data.df[label_col])
        else:
            X_online, y_online = X_test, y_test
 

        online_result = run_online(models, X_online, y_online, rate, n_steps, seed, current_mode, label_names)
        log_manager.log_results(online_result, f"{current_mode}_online")
 
    print("\n✓ Pipeline complete.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NIDS -- ML-based Intrusion Detection')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Training dataset CSV (overrides constants.py)')
    parser.add_argument('--online-dataset', type=str, default=None,
                        dest='online_dataset',
                        help='Separate CSV for the online simulation phase '
                             '(e.g. a single day). If omitted, uses the test '
                             'split from --dataset.')
    parser.add_argument('--mode',    type=str,   default=None,
                        choices=['binary', 'multiclass', 'both'])
    parser.add_argument('--rate',    type=float, default=None,
                        help='Poisson lambda -- avg flows/step')
    parser.add_argument('--steps',   type=int,   default=None,
                        help='Number of simulation steps')
    parser.add_argument('--seed',    type=int,   default=None)
    args = parser.parse_args()

    run(
        dataset_path=args.dataset,
        online_dataset_path=args.online_dataset, 
        mode=args.mode,
        rate=args.rate,
        n_steps=args.steps,
        seed=args.seed,
    )
