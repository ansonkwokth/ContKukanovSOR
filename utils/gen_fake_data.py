import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import logging
from utils.logger import get_logger
import argparse

logger = get_logger('fake_data_generator', logging.INFO)

def generate_fake_data(filepath: str, output_path: str):
    logger.info(f"Loading data from {filepath} ...")
    df = pd.read_csv(filepath)

    # Sort to ensure the last occurrence is at the bottom of the group
    df = df.sort_values(['ts_event', 'publisher_id'])

    # Group by ts_event
    def modify_group(group):
        duplicated = group.duplicated(subset='publisher_id', keep=False)
        if duplicated.any():
            # Get the last duplicated row and change publisher_id to -1
            dup_rows = group[duplicated]
            last_idx = dup_rows.index[-1]
            group.loc[last_idx, 'publisher_id'] = -1
        return group

    df_modified = df.groupby('ts_event', group_keys=False).apply(modify_group)

    df_modified.to_csv(output_path, index=False)
    logger.info(f"Saved fake data to {output_path} ...")



def parse_args():
    parser = argparse.ArgumentParser(description="Fake Data Generator")
    parser.add_argument("-i", "--ifile", type=str, required=True, help="Path to L1 message CSV file")
    parser.add_argument("-o", "--ofile", type=str, required=True, help="Path output fake L1 message CSV file")
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_args()

    generate_fake_data(args.ifile, args.ofile)

