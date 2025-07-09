import yaml
import pandas as pd
import pyarrow


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DataProcessor:
    def __init__(self, config):
        self.config = config

    def load_dataset(self, proportion_w_data=0.8):
        dataframe = pd.read_parquet(self.config["DATAFRAME_PATH"])
        features_to_use_df = pd.read_csv(self.config["FEATURES_TO_USE_PATH"])
        # ... existing logic ...
        return dataframe, features_to_use_df

    def split_train_test(self, dataframe, include_val=False, val_size=0., test_size=0.2):
        # ... existing logic ...
        pass

    def split_into_rl_dataset(self, dataframe):
        # ... existing logic ...
        pass



def main():
    hospital = 'choc'  # or 'mimic'
    config_path = f"config/{hospital}.yaml"
    config = load_config(config_path)

    processor = DataProcessor(config)
    dataframe, features_to_use_df = processor.load_dataset(proportion_w_data=0.8)
    train_data, test_data = processor.split_train_test(dataframe, include_val=False)
    dataset = processor.split_into_rl_dataset(dataframe)
    
    pass

if __name__ == "__main__":
    main()