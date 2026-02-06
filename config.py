import os
import torch

class Config:
    # --- 1. PATHS (Cloud & Cross-Platform Friendly) ---
    # We use os.path.dirname to get the folder where this script lives
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # These paths now work on Windows AND Linux automatically
    DATA_ROOT = os.path.join(BASE_DIR, "testing")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOG_DIR = os.path.join(BASE_DIR, "checkpoints", "logs")
    
    # --- 2. HARDWARE ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Cloud Run CPUs are virtual, so 0 is safest to avoid hanging
    NUM_WORKERS = 0  
    
    PIN_MEMORY = True

    # --- 3. DATA SPECS ---
    TARGET_SPACING = [1.0, 1.0, 1.0]
    PATCH_SIZE = [128, 128, 128] 
    IN_CHANNELS = 4
    NUM_CLASSES = 4 

    # --- 4. ARCHITECTURE ---
    EMBEDDING_DIM = 96
    DEPTHS = [2, 2, 2, 2]
    NUM_HEADS = [3, 6, 12, 24]
    WINDOW_SIZE = [4, 4, 8, 4]

    # --- 5. TRAINING ---
    BATCH_SIZE = 1  
    ACCUM_STEPS = 4
    
    MAX_EPOCHS = 100 
    BASE_LR = 3e-4
    WEIGHT_DECAY = 1e-5
    DEEP_SUPERVISION_WEIGHTS = [0.5, 0.25, 0.125]

    # --- 6. INFERENCE ---
    VAL_INTERVAL = 2
    SW_BATCH_SIZE = 2 
    INFERENCE_OVERLAP = 0.5