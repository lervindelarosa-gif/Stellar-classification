"""Load and clean the SDSS17 Stellar Classification dataset."""
import os

import kagglehub
import pandas as pd

DATASET = "fedesoriano/stellar-classification-dataset-sdss17"
FILE_NAME = "star_classification.csv"

# Bookkeeping columns: they describe how the telescope recorded an object, not the object itself
ID_COLUMNS = ["obj_ID", "run_ID", "rerun_ID", "cam_col", "field_ID",
              "spec_obj_ID", "plate", "MJD", "fiber_ID"]
BRIGHTNESS_COLUMNS = ["u", "g", "r", "i", "z"]


def use_system_certificates():
    """Use the Windows certificate store for downloads.

    Fixes SSL errors when antivirus software (e.g. Avast) inspects HTTPS traffic.
    """
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


def load_raw():
    """Download the dataset from Kaggle (cached after the first time) and return it as a DataFrame."""
    use_system_certificates()
    path = kagglehub.dataset_download(DATASET)
    return pd.read_csv(os.path.join(path, FILE_NAME))


def clean(df):
    """Drop the ID columns and rows with placeholder brightness values (-9999)."""
    data = df.drop(columns=ID_COLUMNS)
    valid = (data[BRIGHTNESS_COLUMNS] > -1000).all(axis=1)
    return data[valid].copy()
