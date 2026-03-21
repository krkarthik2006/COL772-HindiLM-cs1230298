import math


def calculate_bpc(avg_loss: float, total_tokens: int, total_chars: int) -> float:
    """
    Compute Bits Per Character (BPC).
    Converts token-level natural log loss to character-level base-2 bit loss.
    """
    # 1. Get the absolute total loss across the dataset
    total_loss = avg_loss * total_tokens
    
    # 2. Distribute over characters and convert from base 'e' to base 2
    bpc = total_loss / (total_chars * math.log(2))
    return bpc