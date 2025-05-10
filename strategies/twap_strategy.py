import pandas as pd
from datetime import timedelta

def det_buckets(start, end, size=60):
    """
    Generate a list of bucket start times between `start` and `end`,
    spaced by `size` seconds. Handles nanosecond ISO 8601 strings.
    Parameters:
    - start (str): ISO8601 timestamp with nanoseconds
    - end (str): ISO8601 timestamp with nanoseconds
    - size (int): bucket size in seconds (default = 60)
    Returns:
    - List of pandas.Timestamp bucket start times
    """
    start_time = pd.to_datetime(start)
    end_time = pd.to_datetime(end)
    bucket_size = timedelta(seconds=size)

    buckets = []
    current = start_time
    while current < end_time:
        buckets.append(current)
        current += bucket_size

    return buckets


def find_bucket(time, buckets, size):
    time = pd.to_datetime(time)

    bucket_size = timedelta(seconds=size)

    for start in buckets:
        end = start + bucket_size
        if start <= time < end:
            return start

    return None


def twap_strategy(bucket_order_size, venues, buckets, size, ts):
    ts_time = pd.to_datetime(ts)
    bucket = find_bucket(ts_time, buckets, size)
    bucket_order_size_now = bucket_order_size[bucket]

    total_size = sum(venue.ask_size for venue in venues)
    allocation = [0] * len(venues)
    if total_size == 0 or bucket_order_size_now == 0:
        return allocation, bucket_order_size

    sorted_indices = [i for i, _ in sorted(enumerate(venues), key=lambda x: x[1].ask)]

    for idx in sorted_indices:
        v = venues[idx]
        exe_size = min(v.ask_size, bucket_order_size_now)
        allocation[idx] = exe_size
        bucket_order_size_now -= exe_size

    bucket_order_size[bucket] = bucket_order_size_now


    return allocation, bucket_order_size







