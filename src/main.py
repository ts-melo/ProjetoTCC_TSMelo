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
import time

#training the models
def run_offline(data, current_mode, choice_model=None): 
    print(f"\n{'='*60}")
    print(f"  OFFLINE TRAINING — {current_mode.upper()}")
    if choice_model:
        print(f"  Selected model: {choice_model}")
    print(f"{'='*60}")

    data.prepare(mode=current_mode)
    X_train, X_test, y_train, y_test = data.get_split()
    label_names = data.label_names() if current_mode == 'multiclass' else ['BENIGN', 'ATTACK']

    models = ModelManager()
    models.train_all(X_train, y_train, only = choice_model)

    models.evaluate_all(X_test, y_test, mode=current_mode, label_names=label_names)
    if len(models.models) > 1:
        models.compare()

    return models, X_test, y_test, label_names

def run_online(models, X_online, y_online, rate, n_steps, seed, current_mode, label_names, choice_model=None):

    print(f"\n{'='*60}")
    print(f"  ONLINE SIMULATION — {current_mode.upper()}")
    print(f"  lambda={rate} flows/step  |  {n_steps} steps")
    print(f"{'='*60}")

    models_to_run = {
        name: clf for name, clf in models.models.items()
        if choice_model is None or name == choice_model
    }
    print(f"\n[Main] Models selected for online simulation: {', '.join(models_to_run.keys())}")

    print(f"\n[Main] Generating online traffic with Poisson rate λ={rate}")
    tasks = TaskManager(rate=rate, seed=seed)
    tasks.load_flows(X_online, y_online, feature_names=[])
    tasks.generate_arrivals(n_steps)

    # todos os modelos vão ser testados com o mesmo batch
    batches = []
    for step in range(n_steps):
        arrived = tasks.step()
        batch = tasks.drain_pending()
        batches.append((step, arrived, batch))
    
    total_flows = sum(len(b) for _, _, b in batches)
    print(f"\n[Main] {len(batches)} steps - Total flows generated: {total_flows:,}")
    

    all_results = {}

    for classifier_name, classifier in models_to_run.items():
        print(f"\n{'-'*60}")
        print(f"\n[Main] Running online simulation with {classifier_name}")
        print(f"\n{'-'*60}")

        attack_counts = defaultdict(int)
        step_log = []
        sim_start = time.time()


        for step, arrived, batch in batches:
            if not batch:
                continue

            X_batch = np.array([f['features'] for f in batch])
            flow_ids = [f['flow_id'] for f in batch]
            true_labels = [f['true_label'] for f in batch]
            predictions = classifier.predict(X_batch)

            n_attacks = 0
            step_attack_types = defaultdict(int)
            for pred in predictions:
                pred_int = int(pred)
                if pred_int != 0:  
                    n_attacks += 1
                    label_str = label_names[pred_int] if pred_int < len(label_names) else f"Class {pred_int}"
                    step_attack_types[pred_int] += 1
                    attack_counts[label_str] += 1

            n_correct = int((predictions == true_labels).sum())
            step_log.append({
                'step': step,
                'arrived': len(arrived),
                'classified': len(batch),
                'attacks_detected': n_attacks,
                'correct': n_correct,
                'attack_types': dict(step_attack_types),
                'cpu_time' : time.time() - sim_start
            })

            if step % max(1, n_steps // 10) == 0:
                print(f"  Step {step:3d} - Arrived: {len(arrived):4d} | "
                      f"Classified: {len(batch):4d} | "
                      f"Attacks Detected: {n_attacks:4d} |"
                      f"Correct: {n_correct:4d}")
                
        sim_time = round(time.time() - sim_start, 3)
        total_classified = sum(s['classified'] for s in step_log)
        total_attacks = sum(s['attacks_detected'] for s in step_log)
        total_correct = sum(s['correct'] for s in step_log)
        accuracy = round(total_correct / total_classified, 4) if total_classified else 0

        print(f"\n Total flows classified: {total_classified:,} |"
              f" Total attacks detected: {total_attacks:,} | "
              f" Overall accuracy: {accuracy:.4f} |"
              f" Simulation time: {sim_time}s")
        
        if attack_counts:
            print(f"\n Attack type distribution:")
            for label, count in sorted(attack_counts.items(), key=lambda x: x[1]):
                pct = count / total_attacks * 100 if total_attacks else 0
                print(f"{label:<35} {count:>6,} ({pct:5.1f}%)")
        else:
            print("\n No attacks detected in the simulation.")

        all_results[classifier_name] = {
            'total_classified': total_classified,
            'total_attacks': total_attacks,
            'overall_accuracy': accuracy,
            'simulation_time_s': sim_time,
            'lambda': rate,
            'n_steps': n_steps,
            'attack_counts': dict(attack_counts),
            'steps': step_log,
        }

    #tabela
    print(f"\n{'='*60}")
    print(f"  ONLINE SIMULATION SUMMARY")
    print(f"{'='*60}")
    header = f"{'Model':<35} {'Accuracy':>10} {'Classified':>10} {'Attacks':>10} {'Sim Time(s)':>12}"
    print(header)
    print(" " + "─" * (len(header)-2))
    for name, r in all_results.items():
        print(f"{name:<35} "
              f"{r['overall_accuracy']:>10.4f} "
              f"{r['total_classified']:>10,} "
              f"{r['total_attacks']:>10,} "
              f"{r['simulation_time_s']:>12.3f}")
        print(" " + "─" * (len(header)-2))

 
    return all_results


def run(dataset_path=None, online_dataset_path=None, mode=None, rate=None, n_steps=None, seed=None, choice_model=None):

    mode    = mode    or CONSTANTS.CLASSIFICATION_MODE
    rate    = rate    or CONSTANTS.POISSON_RATE
    n_steps = n_steps or CONSTANTS.SIM_STEPS
    seed    = seed    or CONSTANTS.RANDOM_STATE
    online_dataset_path = online_dataset_path or CONSTANTS.ONLINE_DATASET_FILE
    choice_model = choice_model or CONSTANTS.ONLINE_MODEL
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

        models, X_test, y_test, label_names = run_offline(data, current_mode, choice_model)
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
 

        online_result = run_online(models, X_online, y_online, rate, n_steps, seed, current_mode, label_names, choice_model=choice_model)
        
        for model_name, model_result in online_result.items():
            log_manager.log_results(model_result, f"{current_mode}_online", model_name=model_name)
    
 
    print("\n Pipeline complete.\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NIDS -- ML-based Intrusion Detection')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Training dataset CSV (overrides constants.py)')
    parser.add_argument('--online-dataset', type=str, default=None,
                        dest='online_dataset',
                        help='Separate CSV for the online simulation phase '
                             '(e.g. a single day). If omitted, uses the test '
                             'split from --dataset.')
    parser.add_argument('--model', type=str, default=None,
                        dest='choice_model',
                        choices=['decision_tree', 'random_forest', 'mlp'],
                        help='Model to train and use in online phase. If omitted, runs all.')
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
        choice_model=args.choice_model
    )
