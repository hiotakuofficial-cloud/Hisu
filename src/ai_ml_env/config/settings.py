from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

for path in (DATA_DIR, MODELS_DIR, LOGS_DIR):
    path.mkdir(parents=True, exist_ok=True)

DEFAULT_RANDOM_STATE = 42
TRAIN_TEST_SPLIT = 0.2
