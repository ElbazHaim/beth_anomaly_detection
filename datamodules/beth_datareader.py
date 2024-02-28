import pandas as pd


def read_data() -> set[pd.DataFrame]:
    """
    Read in the data from the csv files.
    Args:
        None

    Returns:
        train_df (pd.DataFrame): The training data.
        val_df (pd.DataFrame): The validation data.
        test_df (pd.DataFrame): The testing data.
    """
    train_df = pd.read_csv("data/labelled_training_data.csv")
    val_df = pd.read_csv("data/labelled_validation_data.csv")
    test_df = pd.read_csv("data/labelled_testing_data.csv")
    return train_df, val_df, test_df
