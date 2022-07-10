# info: https://www.theairlinepilots.com/flightplanningforairlinepilots/notamdecode.php
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
root = os.path.dirname(parent)

from functions_.create_tfr_train_dataset import create_tfr_train_dataset

def main():
    
    CREATE_TRAIN_SET = False

    if CREATE_TRAIN_SET:
        create_tfr_train_dataset()


if __name__ == "__main__":
    main()
