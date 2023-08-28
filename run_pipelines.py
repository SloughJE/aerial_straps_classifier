import sys
import argparse
import yaml
from src.data.label import run_labeling
from src.data.video_processing import reduce_video_size
from src.features.make_features import extract_landmarks_and_features, combine_csv_files
from src.models.train_model import train_model_pipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reduce_quality",
        help="reduce video quality for quicker labeling",
        action="store_true"
    )

    parser.add_argument(
        "--label_data",
        help="label raw data",
        action="store_true"
    )

    parser.add_argument(
        "--make_features",
        help="make features",
        action="store_true"
    )

    parser.add_argument(
        "--combine_feature_csv",
        help="combine feature csv",
        action="store_true"
    )

    parser.add_argument(
        "--train_model",
        help="split train test, and train model",
        action="store_true"
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        print("No arguments, please add arguments")
    else:
        with open("params.yaml") as f:
            params = yaml.safe_load(f)

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

        if args.combine_feature_csv:
            combine_csv_files(
                params['features']
            )

        if args.train_model:
            train_model_pipeline(
                params['model']
            )