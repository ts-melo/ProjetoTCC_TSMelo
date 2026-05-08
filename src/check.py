from data_manager import DataManager

DATASET_PATH = "C:/Users/thais/OneDrive/Documents/Thesis/data/merged_dataset.csv"

data = DataManager()
data.load(DATASET_PATH)
data.clean()

print("\n=== LABELS NO DATASET ===")
labels = sorted(data.df['Label'].unique())
for l in labels:
    print(l)

print("\n=== DISTRIBUIÇÃO ===")
counts = data.df['Label'].value_counts()
total = len(data.df)

for label, count in counts.items():
    pct = (count / total) * 100
    print(f"{label:<20} {count:>10} ({pct:5.2f}%)")

data.prepare(mode='multiclass')

print("\n=== MAPEAMENTO LABEL -> NÚMERO ===")
for label, encoded in zip(
    data.label_encoder.classes_,
    data.label_encoder.transform(data.label_encoder.classes_)
):
    print(f"{label:<20} -> {encoded}")

