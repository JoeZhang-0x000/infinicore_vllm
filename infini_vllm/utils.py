
printed_keys = {}
def print_once(msg: str):
    if msg not in printed_keys:
        printed_keys[msg] = True
        print(msg)
