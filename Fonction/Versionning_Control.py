import importlib.metadata
import subprocess
import sys
import os

Version_torch_GPU = "cu130"

def Version_Control():
    try:
        importlib.metadata.version("torch")
        torch_present = True
    except importlib.metadata.PackageNotFoundError:
        print("Torch absent")
        torch_present = False

    if torch_present:
        import torch
            
        if  (not torch.cuda.is_available() and "--deja-tente" not in sys.argv) or (not torch.__version__[-5:] == Version_torch_GPU and "--deja-tente" not in sys.argv):
            print("torch GPU absent")
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "-y", "torchvision"])
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/" + Version_torch_GPU])
            os.execv(sys.executable, [sys.executable] + sys.argv + ["--deja-tente"])
        else:
            print("torch GPU présent")
            print("Fin de la verification de version")
            print("\n\n\n")

Version_Control()