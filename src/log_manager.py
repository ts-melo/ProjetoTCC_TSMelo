
import os
import json
from datetime import datetime

import utils.constants as CONSTANTS


def _ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def log_results(results: dict, mode: str):
    
    _ensure_dir(CONSTANTS.LOG_FOLDER)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = CONSTANTS.LOG_FOLDER + f"results_{mode}_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"[LogManager] Results saved → {filename}")


def log_summary(results: dict, mode: str):
   
    _ensure_dir(CONSTANTS.OUTPUT_FOLDER)
    filename = CONSTANTS.OUTPUT_FOLDER + "summary.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, 'a') as f:
        f.write(f"\n[{timestamp}] mode={mode}\n")
        for model_key, metrics in results.items():
            line = (
                f"  {model_key:<35} "
                f"acc={metrics['accuracy']:.4f}  "
                f"f1={metrics['f1_score']:.4f}  "
                f"recall={metrics['recall']:.4f}  "
                f"inference={metrics['inference_time_s']:.4f}s\n"
            )
            f.write(line)
    print(f"[LogManager] Summary appended → {filename}")
