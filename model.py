import sys
import os

# --- PATH FIX FOR CLOUD RUN ---
# This tells Python to look inside the "nnFormer" folder to find the "nnformer" package
sys.path.append(os.path.join(os.path.dirname(__file__), 'nnFormer'))
# ------------------------------

import torch
import torch.nn as nn
# Now this import will work because we added the folder above to the path
from nnformer.network_architecture.nnFormer_tumor import nnFormer

# The rest of your model code remains the same (if you have custom wrapper code here, keep it)
# If this file just imports the model, you are done.