import time
import numpy as np
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.neural_network import MLPClassifier

import utils.constants as CONSTANTS


class ModelManager:

    def __init__(self):
        self.results    = {}
        self.models     = {}

    def _build_model(self, name):
        if name == 'decision_tree':
            return DecisionTreeClassifier(
                max_depth=CONSTANTS.DT_MAX_DEPTH,
                random_state=CONSTANTS.RANDOM_STATE
            )
        elif name == 'random_forest':
            return RandomForestClassifier(
                n_estimators=CONSTANTS.RF_N_ESTIMATORS,
                max_depth=CONSTANTS.RF_MAX_DEPTH,
                random_state=CONSTANTS.RANDOM_STATE,
                n_jobs=-1
            )
        elif name == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=CONSTANTS.MLP_HIDDEN_LAYERS,
                max_iter=CONSTANTS.MLP_MAX_ITER,
                activation=CONSTANTS.MLP_ACTIVATION,
                random_state=CONSTANTS.RANDOM_STATE,
                early_stopping=True,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown model: {name}")
        
    def _model_path(self, name, mode):
        os.makedirs(CONSTANTS.MODELS_FOLDER, exist_ok=True)
        return CONSTANTS.MODELS_FOLDER + f"{name}_{mode}.pkl"
    
    def save(self, name, mode):
        if name not in self.models:
            print(f"[ModelManager] No model named '{name}' trained.")
            return
        path = self._model_path(name, mode)
        joblib.dump(self.models[name], path)
        print(f"[ModelManager] Saved {name} to {path}")
    
    def save_all(self, mode):
        for name in self.models:
            self.save(name, mode)
    
    def loaf(self, name, mode):
        path = self._model_path(name, mode)
        if not os.path.exists(path):
            print(f"[ModelManager] No saved model at {path}")
            return False
        self.models[name] = joblib.load(path)
        print(f"[ModelManager] Loaded {name} from {path}")
        return True
    
    def load_all(self, mode):
        loaded = [self.loaf(name, mode) for name in CONSTANTS.MODELS]
        return all(loaded)

    def train(self, name, X_train, y_train):

        print(f"\n[ModelManager] Training {name}...")
        model = self._build_model(name)

        start = time.time()
        model.fit(X_train, y_train)
        elapsed = round(time.time() - start, 3)

        self.models[name] = model
        print(f"[ModelManager] {name} trained in {elapsed}s")
        return model

    def train_all(self, X_train, y_train, only=None):
        if only is None:
            selected = CONSTANTS.MODELS
        else:
            selected = [only] 
        
        for name in selected:
            self.train(name, X_train, y_train)
            
        return self

    def evaluate(self, name, X_test, y_test, mode='binary', label_names=None):
        model = self.models[name]
        start = time.time()
        y_pred = model.predict(X_test)
        inference_time = round(time.time() - start, 4)
        avg = 'binary' if mode == 'binary' else 'weighted'

        metrics = {
            'accuracy':         round(accuracy_score(y_test, y_pred), 4),
            'precision':        round(precision_score(y_test, y_pred, average=avg,     zero_division=0), 4),
            'recall':           round(recall_score(y_test, y_pred,    average=avg,     zero_division=0), 4),
            'f1_weighted':      round(f1_score(y_test, y_pred,        average='weighted', zero_division=0), 4),
            'f1_macro':         round(f1_score(y_test, y_pred,        average='macro',    zero_division=0), 4),
            'f1_micro':         round(f1_score(y_test, y_pred,        average='micro',    zero_division=0), 4),
            'inference_time_s': inference_time,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }

        target_names = label_names if label_names else None
        report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)

        key = f"{name}_{mode}"
        self.results[key] = metrics

        print(f"\n── Results: {name} [{mode}] ──────────────────────────")
        for k, v in metrics.items():
            if k != 'confusion_matrix':
                print(f"  {k:<22}: {v}")
        print(f"\n{report}")

        return metrics

    def evaluate_all(self, X_test, y_test, mode='binary', label_names=None):

        for name in self.models:
            self.evaluate(name, X_test, y_test, mode=mode, label_names=label_names)
        return self

    def compare(self):
        if not self.results:
            print("[ModelManager] No results yet.")
            return

        print("\n── Model Comparison ─────────────────────────────────────────")
        header = f"{'Model':<35} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Inference(s)':>13}"
        print(header)
        print("─" * len(header))
        for key, m in self.results.items():
            print(
                f"{key:<35} "
                f"{m['accuracy']:>9.4f} "
                f"{m['precision']:>10.4f} "
                f"{m['recall']:>8.4f} "
                f"{m['f1_weighted']:>8.4f} "
                f"{m['f1_macro']:>9.4f} "
                f"{m['f1_micro']:>9.4f} "
                f"{m['inference_time_s']:>13.4f}"
            )
        print("─" * len(header))

    def get_results(self):
        return self.results
