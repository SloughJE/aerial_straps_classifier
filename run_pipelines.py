import sys
import argparse
import yaml
from src.data.label import run_labeling, apply_mirror_labels
from src.data.process_media import process_media
from src.features.make_features import extract_landmarks_and_features, extract_landmarks_and_features_for_photos, combine_csv_files
from src.models.train_dev_model import train_model_pipeline
from src.models.train_prod_model import train_prod_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--process_videos", 
        action="store_true", 
        help="Process videos (reduce size for quicker labeling, mirror videos)"
    )
    
    parser.add_argument(
        "--process_photos", 
        action="store_true", 
        help="Process photos (mirror photos)"
    )
 
    parser.add_argument(
        "--label_data",
        help="label raw data",
        action="store_true"
    )

    parser.add_argument(
        "--label_videos",
        help="label videos data",
        action="store_true"
    )

    parser.add_argument(
        "--label_photos",
        help="label photos data",
        action="store_true"
    )

    parser.add_argument(
        "--apply_mirrored_label_data",
        help="label mirrored videos",
        action="store_true"
    )

    parser.add_argument(
        "--make_features_videos",
        help="make features",
        action="store_true"
    )

    parser.add_argument(
        "--make_features_photos",
        help="make features",
        action="store_true"
    )

    parser.add_argument(
        "--combine_feature_csv",
        help="combine feature csv",
        action="store_true"
    )

    parser.add_argument(
        "--train_dev_model",
        help="split train test, and train model",
        action="store_true"
    )

    parser.add_argument(
        "--train_prod_model",
        help="train prod model on entire dataset",
        action="store_true"
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        print("No arguments, please add arguments")
    else:
        with open("params.yaml") as f:
            params = yaml.safe_load(f)

        if args.process_videos:
            process_media(
                params['media_processing'], 'video'
                ) 

        if args.process_photos:
            process_media(
                params['media_processing'], 'photo'
                )
            
        if args.label_videos:
            run_labeling(
                params['labeling'], 'video'
                )

        if args.label_photos:
            run_labeling(
                params['labeling'], 'photo'
                )
        
        if args.apply_mirrored_label_data:
            apply_mirror_labels(
                params['labeling']
            )


        if args.make_features_videos:
            extract_landmarks_and_features(
                params['features']
            )

        if args.make_features_photos:
            extract_landmarks_and_features_for_photos(
                params['features']
            )

        if args.combine_feature_csv:
            combine_csv_files(
                params['features']
            )

        if args.train_dev_model:
            train_model_pipeline(
                params['model_dev']
            )

        if args.train_prod_model:
            train_prod_model(
                params['model_prod']
            )