import matplotlib.pyplot as plt
import numpy as np
from main import run

lambdas = [5, 10, 15]

all_res = {}


for lam in lambdas:
    print(f"\nRunning experiment with lambda={lam}...\n")
    
    res = run(
        rate=lam,
        choice_model=None 
    )
    
    all_res[lam] = res


def plot_results(all_res):
    
    for lam, models in all_res.items():
        model_names = list(models.keys())
        n_models = len(model_names)

        width = 0.25

        # acc
        plt.figure()

        ref_steps = next(iter(models.values()))['steps']
        sampled = ref_steps[::100]

        steps = [s['step'] for s in sampled]
        x = np.arange(len(steps))

        for i, model_name in enumerate(model_names):
            data = models[model_name]['steps'][::100]

            acc = [s['accuracy'] for s in data]

            plt.bar(x + i * width, acc, width, label=model_name)

        plt.xticks(x + width, steps)
        plt.xlabel('Steps (time)')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy over time (λ={lam})')
        plt.legend()
        plt.grid(axis='y')

        plt.show()

        # cpu
        plt.figure()

        for i, model_name in enumerate(model_names):
            data = models[model_name]['steps'][::100]

            cpu = [s['cpu_time'] for s in data]

            plt.bar(x + i * width, cpu, width, label=model_name)

        plt.xticks(x + width, steps)
        plt.xlabel('Steps (time)')
        plt.ylabel('CPU time (s)')
        plt.title(f'CPU time over time (λ={lam})')
        plt.legend()
        plt.grid(axis='y')

        plt.show()


plot_results(all_res)