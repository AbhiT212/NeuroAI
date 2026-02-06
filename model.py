
import sys
import os
# Ensure we can find the installed nnformer
try:
    from nnformer.network_architecture.nnFormer_tumor import nnFormer
    print("✅ Loaded nnFormer from library.")
except ImportError:
    # Fallback if installed in local dir
    sys.path.append(os.path.join(os.getcwd(), 'nnFormer'))
    from nnformer.network_architecture.nnFormer_tumor import nnFormer
    print("✅ Loaded nnFormer from local folder.")
