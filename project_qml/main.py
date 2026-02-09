from main_ablation import main_ablation
from main_debug import main_debug_ablation
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Run in DEBUG mode (minimal execution, no scientific guarantees)"
    )
    return parser.parse_args()

ARGS = parse_args()

DEBUG_MODE = ARGS.debug

if __name__ == "__main__":
    if DEBUG_MODE:
        print("[INFO] Running in DEBUG_MODE=True")
        main_debug_ablation()
    else:
        print("[INFO] Running in FULL / PUBLICATION mode")
        main_ablation()