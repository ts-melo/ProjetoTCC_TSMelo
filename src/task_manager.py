import numpy as np
import pandas as pd
from collections import deque

import utils.constants as CONSTANTS


class TaskManager:

    def __init__(self, rate: float = 10.0, seed: int = 42):
        self.rate           = rate
        self.seed           = seed
        self.rng            = np.random.default_rng(seed)

        self.flow_pool      = []
        self.pool_index     = 0  
        self.arrival_counts = []
        self.pending_queue  = deque()
        self.classified     = {}
        self._step_counter  = 0

    def load_flows(self, X: np.ndarray, y: np.ndarray, feature_names: list):

        print(f"[TaskManager] Loading {len(X):,} flows into pool...")

        y_array = np.array(y)
        self.flow_pool = []
        for i in range(len(X)):
            self.flow_pool.append({
                'flow_id':      f"flow_{i:07d}",
                'features':     X[i],
                'true_label':   int(y_array[i]),
                'status':       'PENDING',
                'arrival_step': None,
                'classified_step': None,
                'predicted_label': None,
            })
        self.pool_index = 0
        print(f"[TaskManager] Flow pool ready — {len(self.flow_pool):,} flows")

    def generate_arrivals(self, n_steps: int):

        self.arrival_counts = self.rng.poisson(
            lam=self.rate, size=n_steps
        ).tolist()

        total = sum(self.arrival_counts)
        print(f"[TaskManager] Poisson arrivals generated:")
        print(f"  steps={n_steps}  λ={self.rate}  "
              f"total_expected={total}  "
              f"avg/step={total/n_steps:.1f}")

        if total > len(self.flow_pool):
            print(f"  ⚠ Expected arrivals ({total:,}) exceed pool size "
                  f"({len(self.flow_pool):,}). Pool will cycle.")

    def step(self) -> list:
        if not self.arrival_counts:
            return []

        n_arrivals = self.arrival_counts.pop(0)
        arrived    = []

        for _ in range(n_arrivals):
            if self.pool_index >= len(self.flow_pool):
                self.pool_index = 0

            flow = self.flow_pool[self.pool_index].copy()
            flow['arrival_step'] = self._step_counter
            flow['status']       = 'SUBMITTED'
            self.pool_index     += 1

            self.pending_queue.append(flow)
            arrived.append(flow)

        self._step_counter += 1
        return arrived

    def drain_pending(self) -> list:

        batch = list(self.pending_queue)
        self.pending_queue.clear()
        return batch

    def peek_pending(self) -> list:
        return list(self.pending_queue)


    def stats(self) -> dict:
        return {
            'pool_size':        len(self.flow_pool),
            'pool_index':       self.pool_index,
            'steps_elapsed':    self._step_counter,
            'pending':          len(self.pending_queue),
            'classified':       len(self.classified),
            'steps_remaining':  len(self.arrival_counts),
        }

    def print_stats(self):
        s = self.stats()
        print(f"\n[TaskManager] Stats @ step {s['steps_elapsed']}")
        print(f"  pending:          {s['pending']:>8,}")
        print(f"  classified:       {s['classified']:>8,}")
        print(f"  steps remaining:  {s['steps_remaining']:>8,}")
        print(f"  pool progress:    {s['pool_index']:>8,} / {s['pool_size']:,}")