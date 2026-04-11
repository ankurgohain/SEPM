from __future__ import annotations
import os 
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
_ENV_FILE = Path(__file__).parent.parent.parent / ".env"
if _ENV_FILE.exists():
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

def _env(key:str, default: Any, cast: callable):
    value = os.getenv(key, default)
    if value is None:
        return default
    try:
        if cast is bool:
            return value in ("1", "true", "yes", "on")
        return cast(value)
    except (ValueError, TypeError):
        return default
    

@dataclass (frozen=True)
class PathConfig:
    root: Path = Path(__file__).parent.parent
    raw_data: Path = field(default_factory=lambda: Path("data/raw"))
    processed_data: Path = field(default_factory=lambda: Path("data/processed"))
    sequences: Path = field(default_factory=lambda: Path("data/sequences"))
    checkpoints: Path = field(default_factory=lambda: Path("artifacts/checkpoints"))
    mlflow_tracking: Path = field(default_factory=lambda: Path("artifacts/mlflow"))
    onnx_export: Path = field(default_factory=lambda: Path("artifacts/onnx"))
    logs: Path = field(default_factory=lambda: Path("logs"))

    def __post_init__(self):
        overrides = {
            "raw_data": "LEARNFLOW_RAW_DATA_PATH",
            "processed_data": "LEARNFLOW_PROCESSED_PATH",
            "checkpoints": "MODEL_CHECKPOINTS_DIR",
            "mlflow_tracking": "MLFLOW_TRACKING_URI",
            "logs": "LEARNFLOW_LOG_PATH"
        }

        for attr, env_key in overrides.items():
            if env_key in os.environ:
                object.__setattr__(self, attr, Path(os.environ[env_key]))

    def make_dir(self) -> None:
        for f in fields(self):
            p = getattr(self, f.name)
            if isinstance(p, Path) and f.name != "root":
                p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:

    numerical_features: tuple = ("quiz_score", "engagement_rate", "hint_count", "session_duration", "correct_attempts", "incorrect_attempts")
    categorical_features: tuple = ("performance_score", 'mastery_achieved', 'dropout_risk')
    seq_len: int = _env("DATA_SEQ_LEN", 10, int)
    mins_sessions: int = _env("DATA_MIN_SESSIONS", 1, int)
    modules: tuple = ("python_basics", 'data_structures', 'ml_foundations', 'deep_learning', 'nlp_basics', 'reinforcement_learning')
    
    train_ratio: float = _env("DATA_TRAIN_RATIO", 0.70, float)
    val_ratio: float = _env("DATA_VAL_RATIO", 0.15, float)
    test_ratio: float = _env("DATA_TEST_RATIO", 0.15, float)

    n_synthetic_learners: int = _env("DATA_N_SYNTHETIC", 600, int)
    synthetic_seed: int = _env("DEEP_SEED", 42, int)

    kafka_bootstrap: str = _env("KAFKA_BOOTSTRAP_SERVERS", "localhost: 9092", str)
    kafka_topic: str = _env("KAFKA_LMS_TOPIC", "lms.evnets", str)
    kafka_group_id: str = _env("KAFKA_GROUP_ID", "learnflow-ingest", str)
    kafka_batch_size: int = _env("KAFKA_BATCH_SIXE", 100, int)

    csv_delimiter: str =_env("CSV_DEELIMITER", ",", str)
    csv_encoding: str=_env("CSV_ENCODING", "utf-8",str)
    ingest_chunk_size:int=_env("INGEST_CHUNK_SIZE",100,int)

    @property
    def n_features(self) -> int:
        return len(self.numerical_features)
    
    @property
    def n_modules(self) -> int:
        return len(self.modules)
    
    @property
    def modules_to_id(self) -> dict[str, int]:
        return {m: i for i,m in enumerate(self.modules)}
    
@dataclass(frozen=True)
class ModelConfig:

    embed_dim:int=_env("MODEL_EMBED_DIM",8,int)
    lstm_units_1:int=_env("MODEL_LSTM1_UNITS",128,int)
    lstm_units_2:int=_env("MODEL_LSTM2_UNITS",64,int)
    attention_units:int=_env("MODEL_ATTN_UNITS",64,int)
    dense_units:int=_env("MODEL_DENSE_UNITS",64,int)
    head_units:int=_env("MODEL_HEAD_UNITS",32,int)

    dropout_rate:float=_env("MODEL_DROPOUT",0.3,float)
    recurrent_dropout:float=_env("MODEL_REC_DROPOUT",0.1,float)
    l2_reg:float=_env("MODEL_L2_REG",1e-4,float)

    loss_weight_performance:float=_env("LOSS_W_PERFORMANCE",0.3,float)
    loss_weight_mastery:float=_env("LOSS_W_MASTERY", 0.35, float)
    loss_weight_dropout:float=_env("LOSS_W_DROPOUT",0.35,float)

    checkpoint_name:str=_env("MODEL_CHECKPOINT_NAME","best_model.keras", str)
    scaler_name:str=_env("MODEL_SCALER_NAME","scaler.npy",str)
    onnx_name:str=_env("MODEL_ONNX_NAME","model.onnx",str)

    version:str=_env("MODEL_VERSION","1.0.0",str)

@dataclass(frozen=True)
class TrainingConfig:
    epochs:int=_env("TRAIN_ERRORS",60,int)
    batch_size:int=_env("TRAIN_BATCH_SIZE",32,int)
    learning_rate:float=_env("TRAIN_LR",1e-3,float)
    lr_min:float=_env("TRAIN_LR_MIN",1e-6,float)
    clipnorm:float=_env("TRAIN_CLIPNORM",1.0,float)

    es_patience: int=_env("TRAIN_ES_PATIENCE",10,int)
    es_monitor:str=_env("TRAIN_ES_MONITOR","val_loss",str)
    restore_best:bool=_env("TRAIN_RESTORE_BEST",True,bool)
    lr_patience:int=_env("TRAIN_LR_PATIENCE",5,int)
    lr_factor:float=_env("TRAIN_LR_FACTOR",0.5,float)

    mlflow_experiments:str=_env("MLFLOW_EXPERIMENT","learnflow-lstm",str)
    log_artifacts:bool=_env("MLFLOW_LOG_ARTIFACTS",True,bool)
    seed:int=_env("TRAIN_SEED",42,int)

@dataclass(frozen=True)
class InferenceConfig:
    risk_threshold_high:float=_env("INF_RISK_HIGH",0.65,float)
    risk_threshold_medium:float=_env("INF_RISK_MEDIUM",0.40,float)
    mastery_threshold_low:float=_env("INF_MASTERY_LOW",0.30,float)
    mastery_threshold_high:float=_env("INF_MASTERY_HIGH",0.75,float)
    perf_threshold_low:float=_env("INF_PERF_LOW",60.0,float)
    batch_size:int=_env("INF_BATCH_SIZE",256, int)
    max_workers:int=_env("INF_MAX_WORKERS",4,int)
    cache_ttl_seconds:int=_env("INF_CACHE_TTL",300,int)
    use_onnx:bool=_env("INF_USE_ONNX",False,bool)
    onnx_providers:tuple=("CPUExecutionProvider",)


@dataclass(frozen=True)
class InterventionConfig:
    alert_dropout_threshold:float=_env("IV_ALERT_DROPOUT",0.65,float)
    remedial_mastery_threshold:float=_env("IV_REMEDIAL_MASTERY",0.40,float)
    remedial_perf_threshold:float=_env("IV_REMEDIAL_PERF",60.0,float)
    nudge_dropout_threshold:float=_env("IV_NUDGE_DROPOUT", 0.40, float)
    badge_mastery_threshold:float=_env("IV_BADGE_MASTERY",0.75,float)

    cooldown_sec:int=_env("IV_COOLDOWN_SEC", 3600, int)
    email_enabled:bool=_env("IV_EMAIL_ENABLED",False, bool)
    webhook_enabled:bool=_env("IV_WEBHOOK_ENABLED",False,bool)
    webhook_url: str=_env("IV_WEBHOOK_URL","", str)

    badge_xp:int=_env("IV_BADGE_XP",100, int)
    badge_names:tuple=("Quick Learner", "Consistent Performer", "Deep THinker", "Module Master", "Streak Champion", "Top Performer")

@dataclass(frozen=True)
class APIConfig:
    host: str=_env("API_HOST","0.0.0.0",str)
    port: int=_env("API_PORT",8000,int)
    reload: bool=_env("API_RELOAD",False,bool)
    workers: int = _env("API_WORKERS",1,int)
    log_level: str=_env("LOG_LEVEL","INFO",str)
    api_key: str=_env("LEARNFLOW_API_KEY", "dev-insecure-key-change-me", str)
    rate_limit_rpm:int= _env("RATE_LIMIT_RPM", 60, int)
    cors_original: str=_env("CORS_ORIGINS", "*", str)
    
    
@dataclass(frozen=True)
class LearnFlowConfig:
    """
        Top-level config object.  Import and use as:
 
        from configs.config import cfg
        cfg.model.lstm_units_1        # → 128
        cfg.training.epochs           # → 60
        cfg.inference.risk_threshold_high  # → 0.65
    """
    paths:        PathConfig        = field(default_factory=PathConfig)
    data:         DataConfig        = field(default_factory=DataConfig)
    model:        ModelConfig       = field(default_factory=ModelConfig)
    training:     TrainingConfig    = field(default_factory=TrainingConfig)
    inference:    InferenceConfig   = field(default_factory=InferenceConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    api:          APIConfig         = field(default_factory=APIConfig)

    def summary(self) -> str:
        lines = ["LearnFlow Configuration", "═" * 48]
        for f in fields(self):
            sub = getattr(self, f.name)
            lines.append(f"\n  [{f.name.upper()}]")
            for sf in fields(sub):
                val = getattr(sub, sf.name)
                if isinstance(val, (tuple, list)) and len(val) > 4:
                    val = f"({len(val)} items)"
                lines.append(f"    {sf.name:<28} = {val}")
        return "\n".join(lines)
cfg = LearnFlowConfig()
if __name__=="__main__":
    print(cfg.summary())