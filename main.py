import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import argparse

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)