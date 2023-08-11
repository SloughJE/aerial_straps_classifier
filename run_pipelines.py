import sys
import argparse
import yaml
from src.data.load_data import load_raw_data
from src.labeling.label import run_labeling
from src.labeling.video_processing import reduce_video_size
from src.features.make_features import extract_landmarks_and_features

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load_data",
        help="load raw data",
        action="store_true"
    )
     
    parser.add_argument(
        "--label_data",
        help="label raw data",
        action="store_true"
    )

    parser.add_argument(
        "--reduce_quality",
        help="reduce video quality for quicker labeling",
        action="store_true"
    )

    parser.add_argument(
        "--make_features",
        help="make features",
        action="store_true"
    )

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

        if args.reduce_quality:
            reduce_video_size(
                params['video_processing']
            )

        if args.label_data:
            run_labeling(
                params['labeling']
            )
        
        if args.make_features:
            extract_landmarks_and_features(
                params['features']
            )