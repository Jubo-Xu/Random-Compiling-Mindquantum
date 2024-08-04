
colors = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "pink": "\033[35m",
        "reset": "\033[0m"
    }

def print_colored(text, color):
    # Get the ANSI code for the given color
    color_code = colors.get(color.lower(), colors["reset"])
    # Print the text with the chosen color
    print(f"{color_code}{text}{colors['reset']}")

# The function to transfer a bitstring into corresponding decimal index
def bitstring_to_decimal_idx(bitstring):
    idx = 0
    for i in range(len(bitstring)):
        idx += int(bitstring[i])*(2**(len(bitstring)-i-1))
    return idx