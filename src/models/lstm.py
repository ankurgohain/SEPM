"""
lstm_model.py
Architecture

Input  → Embedding (categorical) + Dense (numerical)
       → Stacked LSTM (2 layers) with dropout
       → Attention mechanism
       → Multi-task output heads:
           |- performance_score   (regression)
           |- mastery_probability (binary classification)
           |- dropout_risk        (binary classification)

Sample dataset: synthetically generated learner interaction sequences
covering quiz scores, engagement rate, hint usage, badge events, and
session durations across 10 timesteps per learner.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, classification_report
import tensorflow as tf
from keras._tf_keras import keras
from keras import layers, Model, callbacks
import warnings

try:
    register_keras_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_keras_serializable = keras.utils.register_keras_serializable

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

class LearnerDataGenerator:
    """
    Generates synthetic learner interaction sequences that mimic
    a real LMS event log. Each learner has a sequence of T timesteps,
    each timestep capturing one learning session.

    Features per timestep
    ---------------------
    Numerical (6):
        quiz_score        - 0..100, raw quiz score
        engagement_rate   - 0..1,   fraction of session time spent on-task
        hint_count        - 0..10,  number of hints requested
        session_duration  - minutes spent in session (5..120)
        correct_attempts  - 0..20,  number of correct problem attempts
        incorrect_attempts- 0..20,  number of incorrect problem attempts

    Categorical (1):
        module_id         - one of 6 course modules (encoded as int)

    Targets (per learner, predicted at sequence end)
    ------------------------------------------------
        performance_score   - float  [0, 100]
        mastery_achieved    - binary {0, 1}
        dropout_risk        - binary {0, 1}
    """

    MODULES = ["python_basics", "data_structures", "ml_fundamentals",
               "deep_learning", "nlp_basics", "reinforcement_learning"]

    def __init__(self, n_learners: int = 500, seq_len: int = 10, seed: int = 42):
        self.n_learners = n_learners
        self.seq_len = seq_len
        np.random.seed(seed)

    def _learner_profile(self):
        """Sample a latent learner ability in [0,1] that drives all features."""
        ability = np.clip(np.random.beta(2, 2), 0.05, 0.95)
        risk    = 1 - ability + np.random.normal(0, 0.15)
        risk    = np.clip(risk, 0, 1)
        return ability, risk

    def generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Returns
        -------
        X_num   : (n_learners, seq_len, 6)  - numerical features, scaled [0,1]
        X_cat   : (n_learners, seq_len, 1)  - module_id integers
        y       : (n_learners, 3)           - [performance, mastery, dropout]
        raw_df  : flat DataFrame of all events (for inspection)
        """
        records = []
        all_num, all_cat, all_y = [], [], []

        for learner_id in range(self.n_learners):
            ability, base_risk = self._learner_profile()
            seq_num = []
            seq_cat = []
            module_idx = np.random.randint(0, len(self.MODULES))

            for t in range(self.seq_len):
                # Scores trend upward for high-ability learners
                quiz = np.clip(
                    ability * 80 + np.random.normal(0, 12) + t * ability * 1.5,
                    10, 100
                )
                engagement = np.clip(ability * 0.8 + np.random.normal(0, 0.12), 0.05, 1.0)
                hints      = int(np.clip(np.random.poisson((1 - ability) * 5), 0, 10))
                duration   = np.clip(np.random.normal(40 + ability * 30, 10), 5, 120)
                correct    = int(np.clip(np.random.poisson(ability * 15), 0, 20))
                incorrect  = int(np.clip(np.random.poisson((1 - ability) * 8), 0, 20))

                # Module can advance every few sessions
                if t > 0 and t % 4 == 0:
                    module_idx = min(module_idx + 1, len(self.MODULES) - 1)

                seq_num.append([quiz, engagement, hints, duration, correct, incorrect])
                seq_cat.append([module_idx])

                records.append({
                    "learner_id":       learner_id,
                    "timestep":         t,
                    "module":           self.MODULES[module_idx],
                    "quiz_score":       round(quiz, 2),
                    "engagement_rate":  round(engagement, 3),
                    "hint_count":       hints,
                    "session_duration": round(duration, 1),
                    "correct_attempts": correct,
                    "incorrect_attempts": incorrect,
                    "ability_latent":   round(ability, 3),
                })

            # Targets derived from final ability + sequence stats
            last_scores   = [r[0] for r in seq_num[-3:]]
            perf_score    = float(np.mean(last_scores) + np.random.normal(0, 3))
            perf_score    = float(np.clip(perf_score, 0, 100))
            mastery       = int(ability > 0.55 and np.mean(last_scores) > 65)
            dropout       = int(base_risk > 0.55 and np.mean([r[1] for r in seq_num[-3:]]) < 0.4)

            all_num.append(seq_num)
            all_cat.append(seq_cat)
            all_y.append([perf_score, mastery, dropout])

        X_num = np.array(all_num, dtype=np.float32)   # (N, T, 6)
        X_cat = np.array(all_cat, dtype=np.int32)     # (N, T, 1)
        y     = np.array(all_y,  dtype=np.float32)    # (N, 3)

        # Scale numerical features to [0, 1]
        N, T, F = X_num.shape
        flat = X_num.reshape(-1, F)
        scaler = MinMaxScaler()
        flat_scaled = scaler.fit_transform(flat)
        X_num = flat_scaled.reshape(N, T, F).astype(np.float32)

        raw_df = pd.DataFrame(records)
        return X_num, X_cat, y, raw_df, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 2. ATTENTION LAYER
# ─────────────────────────────────────────────────────────────────────────────
@register_keras_serializable(package="Custom", name="BahdanauAttention")
class BahdanauAttention(layers.Layer):
    """
    Additive (Bahdanau) attention over LSTM hidden states.

    Given encoder outputs h_1 … h_T it learns a context vector
    c = Σ α_t * h_t where α_t = softmax(score(h_t)).

    This makes the model interpretable: α_t reveals which
    timesteps (learning sessions) the model weights most heavily
    when predicting dropout or mastery.
    """

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1,     use_bias=False)

    def call(self, encoder_outputs):
        # encoder_outputs: (batch, T, hidden_dim)
        score = self.V(tf.nn.tanh(self.W(encoder_outputs)))  # (batch, T, 1)
        weights = tf.nn.softmax(score, axis=1)               # (batch, T, 1)
        context = tf.reduce_sum(weights * encoder_outputs, axis=1)  # (batch, hidden_dim)
        return context, tf.squeeze(weights, axis=-1)          # (batch, T)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LSTM MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

def build_lstm_model(
    seq_len:        int   = 10,
    num_features:   int   = 6,
    num_modules:    int   = 6,
    embed_dim:      int   = 8,
    lstm_units_1:   int   = 128,
    lstm_units_2:   int   = 64,
    attention_units: int  = 64,
    dropout_rate:   float = 0.3,
    l2_reg:         float = 1e-4,
) -> Model:
    """
    Build the multi-task LSTM model.

    Input branches
    ──────────────
    • num_input  : (batch, seq_len, num_features)   – scaled numerical features
    • cat_input  : (batch, seq_len)                 – module_id integer sequence

    Encoder
    ───────
    • Embedding layer maps module_id → dense vector (embed_dim)
    • Concatenate numerical + embedding features
    • LSTM layer 1 with return_sequences=True
    • LSTM layer 2 with return_sequences=True  (feeds attention)
    • Bahdanau attention → context vector

    Output heads (multi-task)
    ─────────────────────────
    • performance_score   : Dense(1)          linear activation  (regression 0-100)
    • mastery_prob        : Dense(1, sigmoid) binary probability
    • dropout_risk        : Dense(1, sigmoid) binary probability
    """
    reg = keras.regularizers.l2(l2_reg)

    # ── Inputs ──────────────────────────────────────────────────────
    num_input = keras.Input(shape=(seq_len, num_features), name="numerical_input")
    cat_input = keras.Input(shape=(seq_len,),              name="module_id_input")

    # ── Module embedding ─────────────────────────────────────────────
    embedding = layers.Embedding(
        input_dim=num_modules + 1,
        output_dim=embed_dim,
        name="module_embedding"
    )(cat_input)                                          # (batch, T, embed_dim)

    # ── Feature fusion ───────────────────────────────────────────────
    x = layers.Concatenate(name="feature_fusion")([num_input, embedding])
    # → (batch, T, num_features + embed_dim)

    # ── Stacked LSTM ─────────────────────────────────────────────────
    x = layers.LSTM(
        units=lstm_units_1,
        return_sequences=True,
        dropout=dropout_rate,
        recurrent_dropout=0.1,
        kernel_regularizer=reg,
        name="lstm_layer_1"
    )(x)
    x = layers.LayerNormalization(name="layer_norm_1")(x)

    x = layers.LSTM(
        units=lstm_units_2,
        return_sequences=True,
        dropout=dropout_rate,
        recurrent_dropout=0.1,
        kernel_regularizer=reg,
        name="lstm_layer_2"
    )(x)
    x = layers.LayerNormalization(name="layer_norm_2")(x)

    # ── Attention ────────────────────────────────────────────────────
    attention = BahdanauAttention(units=attention_units, name="attention")
    context, attn_weights = attention(x)                  # (batch, lstm_units_2)

    # ── Shared dense ─────────────────────────────────────────────────
    shared = layers.Dense(64, activation="relu",
                          kernel_regularizer=reg, name="shared_dense")(context)
    shared = layers.Dropout(dropout_rate, name="shared_dropout")(shared)

    # ── Output heads ─────────────────────────────────────────────────
    # Head 1: Performance score (regression)
    perf_hidden = layers.Dense(32, activation="relu", name="perf_hidden")(shared)
    performance_output = layers.Dense(1, activation="linear",
                                      name="performance_score")(perf_hidden)

    # Head 2: Mastery probability (classification)
    mastery_hidden = layers.Dense(32, activation="relu", name="mastery_hidden")(shared)
    mastery_output = layers.Dense(1, activation="sigmoid",
                                  name="mastery_prob")(mastery_hidden)

    # Head 3: Dropout risk (classification)
    dropout_hidden = layers.Dense(32, activation="relu", name="dropout_hidden")(shared)
    dropout_output = layers.Dense(1, activation="sigmoid",
                                  name="dropout_risk")(dropout_hidden)

    model = Model(
        inputs=[num_input, cat_input],
        outputs={
            "performance_score": performance_output,
            "mastery_prob":       mastery_output,
            "dropout_risk":       dropout_output,
        },
        name="LearnerProgressionLSTM"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def compile_model(model: Model, learning_rate: float = 1e-3) -> Model:
    """
    Multi-task loss:
      • performance_score  → MeanSquaredError   (normalised by /100)
      • mastery_prob       → BinaryCrossentropy
      • dropout_risk       → BinaryCrossentropy

    Loss weights reflect relative task importance:
    dropout_risk is up-weighted because early detection has higher
    instructional value than precise score regression.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss={
            "performance_score": keras.losses.MeanSquaredError(),
            "mastery_prob":      keras.losses.BinaryCrossentropy(),
            "dropout_risk":      keras.losses.BinaryCrossentropy(),
        },
        loss_weights={
            "performance_score": 0.3,
            "mastery_prob":      0.35,
            "dropout_risk":      0.35,
        },
        metrics={
            "performance_score": [keras.metrics.RootMeanSquaredError(name="rmse")],
            "mastery_prob":      [keras.metrics.AUC(name="auc")],
            "dropout_risk":      [keras.metrics.AUC(name="auc")],
        }
    )
    return model


def get_callbacks(checkpoint_path: str = "artifacts/checkpoints/best_model.keras") -> list:
    return [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model: Model, X_num, X_cat, y_true: np.ndarray, split_name: str = "Test"):
    """
    Print a formatted evaluation report for all three output heads.
    """
    preds = model.predict([X_num, X_cat], verbose=0)

    perf_pred    = preds["performance_score"].squeeze()
    mastery_pred = preds["mastery_prob"].squeeze()
    dropout_pred = preds["dropout_risk"].squeeze()

    perf_true    = y_true[:, 0]
    mastery_true = y_true[:, 1].astype(int)
    dropout_true = y_true[:, 2].astype(int)

    rmse = np.sqrt(mean_squared_error(perf_true, perf_pred))
    try:
        mastery_auc = roc_auc_score(mastery_true, mastery_pred)
        dropout_auc = roc_auc_score(dropout_true, dropout_pred)
    except ValueError:
        mastery_auc = dropout_auc = float("nan")

    mastery_bin = (mastery_pred >= 0.5).astype(int)
    dropout_bin = (dropout_pred >= 0.5).astype(int)

    sep = "─" * 54
    print(f"\n{'═'*54}")
    print(f"  {split_name} Evaluation Results")
    print(f"{'═'*54}")
    print(f"\n  Performance Score (Regression)")
    print(f"  {sep}")
    print(f"  RMSE : {rmse:>8.3f}  (scale 0-100)")
    print(f"  MAE  : {np.mean(np.abs(perf_true - perf_pred)):>8.3f}")

    print(f"\n  Mastery Probability (Classification)")
    print(f"  {sep}")
    print(f"  AUC  : {mastery_auc:>8.4f}")
    print(f"\n{classification_report(mastery_true, mastery_bin, target_names=['Not Mastered','Mastered'])}")

    print(f"\n  Dropout Risk (Classification)")
    print(f"  {sep}")
    print(f"  AUC  : {dropout_auc:>8.4f}")
    print(f"\n{classification_report(dropout_true, dropout_bin, target_names=['Retained','At Risk'])}")
    print(f"{'═'*54}\n")

    return {
        "rmse_performance": rmse,
        "auc_mastery":      mastery_auc,
        "auc_dropout":      dropout_auc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. ATTENTION VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def get_attention_model(trained_model: Model) -> Model:
    """
    Returns a sub-model that also outputs attention weights.
    Useful for explainability: inspect which session (timestep)
    the model focused on for each prediction.
    """
    attn_output = trained_model.get_layer("attention").output  # (context, weights)
    return Model(
        inputs=trained_model.inputs,
        outputs={
            "performance_score": trained_model.output["performance_score"],
            "mastery_prob":      trained_model.output["mastery_prob"],
            "dropout_risk":      trained_model.output["dropout_risk"],
            "attention_weights": attn_output[1],  # (batch, T)
        },
        name="LearnerProgressionLSTM_Explainable"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────

class LearnerPredictor:
    """
    Thin wrapper around the trained model for single-learner
    real-time inference (as called by src/inference/predictor.py).
    """

    RISK_THRESHOLDS = {
        "high":   0.65,
        "medium": 0.40,
        "low":    0.0,
    }

    def __init__(self, model: Model, scaler: MinMaxScaler):
        self.model  = model
        self.scaler = scaler

    def predict(self, num_seq: np.ndarray, cat_seq: np.ndarray) -> dict:
        """
        Parameters
        ----------
        num_seq : (seq_len, 6)  raw (unscaled) numerical features
        cat_seq : (seq_len,)    module_id integers

        Returns
        -------
        dict with performance_score, mastery_prob, dropout_risk,
        dropout_tier, and a recommended intervention string.
        """
        scaled = self.scaler.transform(num_seq)
        X_num  = scaled[np.newaxis]             # (1, T, 6)
        X_cat  = cat_seq[np.newaxis]            # (1, T)

        preds = self.model.predict([X_num, X_cat], verbose=0)
        perf    = float(preds["performance_score"][0, 0])
        mastery = float(preds["mastery_prob"][0, 0])
        dropout = float(preds["dropout_risk"][0, 0])

        tier = "low"
        for t, thresh in self.RISK_THRESHOLDS.items():
            if dropout >= thresh:
                tier = t
                break

        intervention = self._recommend_intervention(mastery, dropout, perf)

        return {
            "performance_score": round(perf, 2),
            "mastery_prob":      round(mastery, 4),
            "dropout_risk":      round(dropout, 4),
            "dropout_tier":      tier,
            "intervention":      intervention,
        }

    @staticmethod
    def _recommend_intervention(mastery, dropout, perf) -> str:
        if dropout > 0.65:
            return "ALERT_INSTRUCTOR"
        if mastery < 0.4 and perf < 60:
            return "ASSIGN_REMEDIAL_MODULE"
        if dropout > 0.4:
            return "SEND_MOTIVATIONAL_NUDGE"
        if mastery > 0.75:
            return "AWARD_BADGE_AND_ADVANCE"
        return "CONTINUE_STANDARD_PATH"


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN – END-TO-END PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 54)
    print("  LearnFlow · LSTM Progression Model")
    print("  Multi-task: Performance · Mastery · Dropout")
    print("═" * 54)

    # ── 8.1 Generate dataset ──────────────────────────────────────────
    print("\n[1/5]  Generating synthetic learner dataset …")
    gen = LearnerDataGenerator(n_learners=600, seq_len=10)
    X_num, X_cat, y, raw_df, scaler = gen.generate()

    print(f"       Learners     : {X_num.shape[0]}")
    print(f"       Sequence len : {X_num.shape[1]} timesteps")
    print(f"       Num features : {X_num.shape[2]}")
    print(f"       Total events : {len(raw_df):,}")
    print(f"       Mastery rate : {y[:, 1].mean():.1%}")
    print(f"       Dropout rate : {y[:, 2].mean():.1%}")

    # Sample of raw events
    print("\n  Sample event log (first 5 rows):")
    display_cols = ["learner_id", "timestep", "module",
                    "quiz_score", "engagement_rate", "hint_count"]
    print(raw_df[display_cols].head().to_string(index=False))

    # ── 8.2 Train / val / test split ─────────────────────────────────
    print("\n[2/5]  Splitting dataset …")
    X_num_tv, X_num_test, X_cat_tv, X_cat_test, y_tv, y_test = train_test_split(
        X_num, X_cat, y, test_size=0.15, random_state=42, stratify=y[:, 2].astype(int)
    )
    X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
        X_num_tv, X_cat_tv, y_tv, test_size=0.15, random_state=42,
        stratify=y_tv[:, 2].astype(int)
    )

    train_targets = {"performance_score": y_train[:, 0],
                     "mastery_prob":      y_train[:, 1],
                     "dropout_risk":      y_train[:, 2]}
    val_targets   = {"performance_score": y_val[:, 0],
                     "mastery_prob":      y_val[:, 1],
                     "dropout_risk":      y_val[:, 2]}

    print(f"       Train : {len(X_num_train)} | Val : {len(X_num_val)} | Test : {len(X_num_test)}")

    # ── 8.3 Build & compile ──────────────────────────────────────────
    print("\n[3/5]  Building model …")
    model = build_lstm_model(seq_len=10, num_features=6, num_modules=6)
    model = compile_model(model, learning_rate=1e-3)
    model.summary(line_length=72)

    # ── 8.4 Train ────────────────────────────────────────────────────
    print("\n[4/5]  Training …")
    history = model.fit(
        x=[X_num_train, X_cat_train],
        y=train_targets,
        validation_data=([X_num_val, X_cat_val], val_targets),
        epochs=60,
        batch_size=32,
        callbacks=get_callbacks(checkpoint_path="/tmp/best_lstm.keras"),
        verbose=1,
    )

    # ── 8.5 Evaluate ─────────────────────────────────────────────────
    print("\n[5/5]  Evaluating on held-out test set …")
    metrics = evaluate_model(model, X_num_test, X_cat_test, y_test, split_name="Test")

    # ── 8.6 Single-learner inference demo ────────────────────────────
    print("  Single-learner Inference Demo")
    print("  " + "─" * 52)
    predictor = LearnerPredictor(model, scaler)

    # Simulate a struggling learner (low scores, high hint usage)
    struggling_seq = np.array([
        [45, 0.3, 7, 25, 3, 12],
        [42, 0.28, 8, 20, 2, 14],
        [40, 0.25, 9, 18, 2, 15],
        [38, 0.22, 9, 15, 1, 16],
        [36, 0.20, 10, 12, 1, 17],
        [35, 0.18, 10, 10, 1, 18],
        [33, 0.15, 10, 8,  1, 18],
        [30, 0.12, 10, 6,  0, 19],
        [28, 0.10, 10, 5,  0, 20],
        [25, 0.08, 10, 4,  0, 20],
    ], dtype=np.float32)
    struggling_modules = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32)

    result = predictor.predict(struggling_seq, struggling_modules)
    print(f"\n  [Struggling Learner]")
    for k, v in result.items():
        print(f"    {k:<22}: {v}")

    # Simulate a high-performing learner
    thriving_seq = np.array([
        [78, 0.82, 1, 55, 14, 3],
        [81, 0.84, 1, 58, 15, 2],
        [83, 0.86, 0, 60, 16, 2],
        [85, 0.87, 1, 62, 16, 1],
        [87, 0.88, 0, 65, 17, 1],
        [89, 0.89, 0, 67, 17, 1],
        [90, 0.90, 0, 70, 18, 1],
        [92, 0.91, 0, 72, 18, 0],
        [93, 0.92, 0, 74, 19, 0],
        [95, 0.93, 0, 75, 20, 0],
    ], dtype=np.float32)
    thriving_modules = np.array([4, 4, 4, 4, 5, 5, 5, 5, 5, 5], dtype=np.int32)

    result2 = predictor.predict(thriving_seq, thriving_modules)
    print(f"\n  [High-performing Learner]")
    for k, v in result2.items():
        print(f"    {k:<22}: {v}")

    print("\n" + "═" * 54)
    print("  Pipeline complete.")
    print("═" * 54 + "\n")

    return model, history, metrics, scaler


if __name__ == "__main__":
    model, history, metrics, scaler = main()
