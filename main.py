import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import argparse
from config import ROOT


DATA_PATH = os.path.join(ROOT, "datasets", "processed", "WLASL100")
ANNOTATION_DIR = os.path.join(ROOT, "datasets", "annotations", "WLASL100")

def default_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_path", default=f"{DATA_PATH}", help="Path to the training dataset")
    parser.add_argument("--annotation_path", default=f"{DATA_PATH}", help="Path to the labels")

    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(args.data_path)
    print(args.annotation_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("", parents=[default_args()], add_help=False)
    args = parser.parse_args()
    main(args)