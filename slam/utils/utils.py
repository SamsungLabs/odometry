def is_int(x: str) -> bool:
    try:
        y = int(x)
        return True
    except ValueError:
        return False
