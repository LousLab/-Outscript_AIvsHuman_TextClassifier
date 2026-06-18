import torch

class CFG:
    # Dataset
    CSV_PATH              = r"C:\Users\rival\Downloads\balanced_ai_human_100k.csv"
    TEXT_COL              = "text"
    LABEL_COL             = "generated"

    # Use ALL data — more variety = less memorization
    MAX_SAMPLES_PER_CLASS = None

    # Model
    MODEL_NAME            = "distilbert-base-uncased"
    MAX_LEN               = 128          # shorter = faster, forces model to use content not structure
    BATCH_SIZE            = 32
    EPOCHS                = 4
    LR                    = 1e-5         # slower learning = less memorization
    WARMUP_RATIO          = 0.15
    VAL_SPLIT             = 0.15
    TEST_SPLIT            = 0.10
    SEED                  = 42

    # Regularization — balanced this time
    DROPOUT               = 0.2
    WEIGHT_DECAY          = 0.05         # stronger L2 penalty to prevent memorization
    LABEL_SMOOTHING       = 0.05         # very mild, just takes edge off overconfidence

    # Early stopping
    PATIENCE              = 2
    MIN_DELTA             = 0.002        # needs meaningful improvement to continue

    # Paths
    SAVE_DIR              = "./distilbert_ai_vs_human"
    GRAPH_DIR             = "./training_graphs"

    # Hardware
    DEVICE                = "cuda" if torch.cuda.is_available() else "cpu"
    FP16                  = torch.cuda.is_available()

    # Prediction threshold — above this = AI, below = Human
    # Set higher than 0.5 to reduce false AI predictions
    AI_THRESHOLD          = 0.55
