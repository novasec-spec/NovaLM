"""
Top-level entry point for NovaLM inference.
Run:
    python generate.py
    python generate.py --prompt "The scientist discovered"
"""

from inference.generate import main

if __name__ == "__main__":
    main()
