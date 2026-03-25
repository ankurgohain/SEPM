import numpy as np 
import pandas as pd 
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, roc_auc_score, classification_report
# import tensorflow as tf
# from keras._tf_keras import keras
# from keras import layers, Model, callbacks
# import warnings
# warnings.filterwarnings("ignore")

class LearnerDataGenerator:

    MODULES = ["python_basics", "data_structures", "ml_fundamentals", "deep_learning", "npm_basics", "reinforcement_learning"]

    def __init__(self, n_learners: int= 500, seq_len: int = 10, seed: int =42):
        self.n_learners = n_learners
        self.seq_len = seq_len
        np.random.seed(seed)

    def _learner_profile(self):
        ability = np.clip(np.random.beta(2,2), 0.05, 0.95)
        risk = 1- ability+np.random.normal(0, 0.15)

        risk = np.clip(risk, 0,1)
        return ability, risk

    def generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        print("Starting data generation...")
        records = []
        all_num, all_cat, all_y = [], [], []

        for learner_id in range(self.n_learners):
            ability, base_risk = self._learner_profile()
            seq_num = []
            seq_cat = []
            module_idx = np.random.randint(0, len(self.MODULES))

            for t in range(self.seq_len):
                quiz = np.clip( ability * 80 + np.random.normal(0,12) + t *ability *1.5, 10, 100)
                engagement = np.clip(ability *0.8 + np.random.normal(0, 0.12), 0.05, 1)
                hints = int(np.clip(np.random.poisson((1-ability) *5), 0, 10))
                duration = np.clip(np.random.normal(40 + ability* 30, 10), 5, 120)
                correct = int(np.clip(np.random.poisson(ability*15),0,20))
                incorrect = int(np.clip(np.random.poisson((1- ability)*8),0,20))

                if t>0 and t%4==0:
                    module_idx = min(module_idx+1, len(self.MODULES)-1)
                    seq_num.append([quiz,engagement,hints,duration,correct,incorrect])
                    seq_cat.append([module_idx])
                    records.append({
                        "learner_id": learner_id,
                        "timestamp": t,
                        "module": self.MODULES[module_idx],
                        "quiz_score": round(quiz, 2),
                        "engagement_rate": round(engagement, 3),
                        "hint_count": hints,
                        "session_duration": round(duration, 1),
                        "correct_attempts": correct,
                        "incorrect_attempts": incorrect,
                        "ability_latent": round(ability, 3),
                    })
        
                last_scores = [r[0] for r in seq_num[-3:]]
                perf_score = float(np.mean(last_scores) +np.random.normal(0,3))
                perf_score = float(np.clip(perf_score,0,100))
                mastery = int(ability>0.55 and np.mean(last_scores)>65)
                dropout = int(base_risk>0.55 and np.mean([r[1] for r in seq_num[-3:]]) < 0.4)
                all_num.append(seq_num)
                all_cat.append(seq_cat)
                all_y.append([perf_score, mastery, dropout])

        X_num = np.array(all_num, dtype=np.float32)
        X_cat = np.array(all_cat, dtype=np.int32)
        y = np.array(all_y, dtype=np.float32)
    
        N,T,F = X_num.shape
        flat = X_num.reshape(-1, F)
        scaler = MinMaxScaler()
        flat_scaled = scaler.fit_transform(flat)
        X_nnum = flat_scaled.reshape(N,T,F).astype(np.float32)
    
        raw_df = pd.DataFrame(records)
        print(f"DataFrame created with {len(records)} records.")
        print("Attempting to save CSV...")
        raw_df.to_csv('C:/Users/Asus/projects/lstm-learner-progression/data/raw/learner_data.csv', index=False)
        print ("CSV saved. Printing other outputs...")

        print (X_num, X_cat, y, raw_df, scaler)
    
LearnerDataGenerator().generate()