# strategies/vwap_liquidity_weighted.py

def vwap_strategy( order_size, venues):
    """
    Allocates shares proportional to available liquidity (ask size),
    and computes VWAP accordingly.

    Args:
        venues (list): List of venue objects with .ask and .ask_size.
        order_size (int): Total number of shares to allocate.

    Returns:
        split (list[int]): Share allocation to each venue.
        cost (float): Estimated VWAP cost of this allocation.
    """
    total_liquidity = sum(v.ask_size for v in venues)

    if total_liquidity == 0:
        return [0] * len(venues), 0.0  # No liquidity available

    # Allocate proportionally to ask size
    raw_allocs = [order_size * (v.ask_size / total_liquidity) for v in venues]
    split = [min(int(round(q)), v.ask_size) for q, v in zip(raw_allocs, venues)]

    # Adjust if rounding errors over- or under-allocated
    diff = order_size - sum(split)
    if diff != 0:
        # Fix under-/overfill by greedily assigning/removing shares from best venues
        sorted_indices = sorted(range(len(venues)), key=lambda i: venues[i].ask)
        for i in sorted_indices:
            if diff == 0:
                break
            if diff > 0:
                # Add share if venue has capacity
                if split[i] < venues[i].ask_size:
                    split[i] += 1
                    diff -= 1
            else:
                # Remove share if nonzero
                if split[i] > 0:
                    split[i] -= 1
                    diff += 1

    # Compute estimated cost (VWAP-style)
    cost = sum(split[i] * venues[i].ask for i in range(len(venues)))

    return split

