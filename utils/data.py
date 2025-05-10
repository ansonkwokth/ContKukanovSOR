import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import logging
from utils.logger import get_logger

logger = get_logger('data_loader', logging.INFO)

def load_and_clean_l1_data(filepath: str) -> list[pd.DataFrame]:
    """
    Load and clean L1 message-by-message market data.

    This function:
    - Reads the CSV file into a DataFrame
    - Filters out rows without a valid ask price
    - Sorts by ts_event
    - Keeps only the first message per venue (publisher_id) per timestamp
    - Groups by ts_event to return per-snapshot data

    Args:
        filepath (str): Path to the L1 data CSV file.

    Returns:
        List[pd.DataFrame]: Each DataFrame is a snapshot of venues at a given ts_event.
    """
    logger.info(f"Loading data from {filepath} ...")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows before filtering.")

    # Keep only rows with valid ask prices (in case of missing ask data)
    df = df[df["ask_px_00"].notnull()]
    logger.info(f"Cleaned to {len(df)} rows after filtering missing ask data.")

    # Sort by ts_event
    df = df.sort_values("ts_event")

    # Keep only the first message per publisher per timestamp
    df = df.drop_duplicates(subset=["ts_event", "publisher_id"], keep="first")

    logger.info(f"Cleaned to {len(df)} rows after keeping only the first message per publisher_id.")

    return list(df.groupby("ts_event"))




