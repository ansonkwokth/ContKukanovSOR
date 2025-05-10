from typing import List, Tuple
from utils.venue import Venue

def clip_split_to_remaining(split: List[int], remaining: int):
    """
    Clips a split list so that its total does not exceed the remaining order size.

    Args:
        split (List[int]): Original allocation suggestion.
        remaining (int): Remaining shares to be filled.

    Returns:
        List[int]: Clipped allocation.
    """
    clipped = []
    total = 0
    for qty in split:
        take = min(qty, max(0, remaining - total))
        clipped.append(take)
        total += take
        if total >= remaining:
            break
    # Pad with zeros to maintain same length
    clipped += [0] * (len(split) - len(clipped))
    return clipped



def cash_spent(split: List[int],
                 venues: List[Venue],
                 order_size: int,
                 no_fee_rebate: bool = True) -> float:
    """
    Computes the total cost for a given allocation (split) across venues.

    Args:
    Returns:
        float: The total cost for the given allocation.
    """
    executed = 0
    cash_spent = 0.0

    for i in range(len(venues)):
        exe = min(split[i], venues[i].ask_size)
        executed += exe
        fee = 0 if no_fee_rebate else venues[i].fee
        rebate = 0 if no_fee_rebate else venues[i].rebate
        cash_spent += exe * (venues[i].ask + fee)
        maker_rebate = max(split[i] - exe, 0) * rebate
        cash_spent -= maker_rebate

    return cash_spent



if __name__ == "__main__":
    print(clip_split_to_remaining([1,1,1,1,1], 100))
    print(clip_split_to_remaining([1,1,1,1,1], 2))
    print(clip_split_to_remaining([1,3,1,1,1], 4))
    print(clip_split_to_remaining([1,3,2,1,1], 5))
