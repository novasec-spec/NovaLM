"""
Top-level entry point for NovaLM training.
Run:
    python train.py
    python train.py --steps 500
    python train.py --reset
"""

from training.train import main

if __name__ == "__main__":
    main()
