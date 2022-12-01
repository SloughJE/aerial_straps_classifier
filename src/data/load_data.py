import pandas as pd

def load_raw_data(input_dir,output_dir):

    df = pd.read_csv(input_dir)

    df.to_csv(output_dir,index=False)
    print(f"data loaded and saved to: {output_dir}")