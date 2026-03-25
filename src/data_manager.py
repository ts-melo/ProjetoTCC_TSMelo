import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import utils.constants as CONSTANTS


class DataManager:


    def __init__(self):
        self.df         = None
        self.X_train    = None
        self.X_test     = None
        self.y_train    = None
        self.y_test     = None
        self.features   = None
        self.label_encoder = LabelEncoder()
        self.scaler     = StandardScaler()


    def load(self, filepath=None):
        path = filepath or CONSTANTS.DATASET_FILE
        print(f"[DataManager] Loading dataset from: {path}")
        self.df = pd.read_csv(path, low_memory=False)

        self.df.columns = self.df.columns.str.strip()
        print(f"[DataManager] Loaded {len(self.df):,} rows × {len(self.df.columns)} columns")
        return self

    def load_multiple_days(self, filepaths: list):
        dfs = []
        for fp in filepaths:
            print(f"[DataManager] Loading: {fp}")
            tmp = pd.read_csv(fp, low_memory=False)
            tmp.columns = tmp.columns.str.strip()
            dfs.append(tmp)
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"[DataManager] Combined dataset: {len(self.df):,} rows")
        return self


    def clean(self):
        before = len(self.df)

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)

        after = len(self.df)
        print(f"[DataManager] Cleaning: {before:,} → {after:,} rows (removed {before - after:,})")
        return self


    def prepare(self, mode='binary'):

        label_col = 'Label'  

        self.features = [c for c in self.df.columns if c != label_col]
        X = self.df[self.features].select_dtypes(include=[np.number])
        self.features = X.columns.tolist()

        if mode == 'binary':
            y_raw = self.df[label_col].apply(
                lambda x: 0 if x == CONSTANTS.BENIGN_LABEL else 1
            )
            print(f"[DataManager] Binary labels  — BENIGN: {(y_raw==0).sum():,}  ATTACK: {(y_raw==1).sum():,}")
        else:
            y_raw = self.label_encoder.fit_transform(self.df[label_col])
            classes = self.label_encoder.classes_
            print(f"[DataManager] Multiclass labels ({len(classes)} classes): {list(classes)}")

        X_scaled = self.scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y_raw,
            test_size=CONSTANTS.TEST_SIZE,
            random_state=CONSTANTS.RANDOM_STATE,
            stratify=y_raw
        )

        print(f"[DataManager] Train: {len(self.X_train):,}  Test: {len(self.X_test):,}")
        return self

    def get_split(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def label_names(self):
        try:
            return list(self.label_encoder.classes_)
        except Exception:
            return ['BENIGN', 'ATTACK']

    def summary(self):
        print("\n── Dataset Summary ──────────────────────────────")
        print(self.df['Label'].value_counts())
        print("─────────────────────────────────────────────────\n")
