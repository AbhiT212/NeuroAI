import torch

class Config:
    # --- 1. PATHS (Windows Format) ---
    # Using raw strings (r"...") handles backslashes correctly
    DATA_ROOT = r"D:\my_project_backup\testing" 
    CHECKPOINT_DIR = r"D:\my_project_backup\checkpoints"
    LOG_DIR = r"D:\my_project_backup\checkpoints\logs"
    
    # --- 2. HARDWARE (WINDOWS SAFE MODE) ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ⚠️ CRITICAL WINDOWS FIX:
    # Windows has issues with high worker counts. 
    # Set to 0 (runs on main thread) to prevent hanging/crashing.
    # You can try 2, but if it freezes, go back to 0.
    NUM_WORKERS = 0  
    
    PIN_MEMORY = True

    # --- 3. DATA SPECS ---
    TARGET_SPACING = [1.0, 1.0, 1.0]
    # Keep patch size high for accuracy, but we will lower batch size to fit it.
    PATCH_SIZE = [128, 128, 128] 
    IN_CHANNELS = 4
    NUM_CLASSES = 4 

    # --- 4. ARCHITECTURE ---
    EMBEDDING_DIM = 96
    DEPTHS = [2, 2, 2, 2]
    NUM_HEADS = [3, 6, 12, 24]
    WINDOW_SIZE = [4, 4, 8, 4]

    # --- 5. TRAINING (Consumer GPU Friendly) ---
    # Reduced from 4 to 1 or 2 to save VRAM on local PC
    BATCH_SIZE = 1  
    ACCUM_STEPS = 4 # Gradient accumulation compensates for small batch size
    
    MAX_EPOCHS = 100 
    BASE_LR = 3e-4
    WEIGHT_DECAY = 1e-5
    DEEP_SUPERVISION_WEIGHTS = [0.5, 0.25, 0.125]

    # --- 6. INFERENCE / VAL ---
    VAL_INTERVAL = 2
    
    # ⚠️ CRITICAL MEMORY FIX:
    # 16 is too high for RTX cards. 
    # Set to 2 or 4. If you get CUDA OOM, set to 1.
    SW_BATCH_SIZE = 2 
    INFERENCE_OVERLAP = 0.5