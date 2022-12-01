import sys 
import yaml
import argparse

from src.data.load_data import load_raw_data
#from src.features.make_features import *
#from src.models import predict_model
#from src.models import train_model
# etc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load_data",
        help="load raw data",
        action="store_true"
    )
     
    parser.add_argument(
        "--make_features",
        help="make features",
        action="store_true"
    )
    # etc

    args = parser.parse_args()

    if len(sys.argv) == 1:
        print("No arguments, please add arguments")
    else:
        with open("params.yaml") as f:
            params = yaml.safe_load(f)

        if args.load_data:
            load_raw_data(
                'data/raw/test.csv',
                'data/interim/processed.csv'
            )

