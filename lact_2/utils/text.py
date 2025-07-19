# file: utils/text.py
# Character mapping for text <-> sequence conversion
# We define a vocabulary: 0 = blank (for CTC), 1-26 = 'A'-'Z', 27 = apostrophe, 28 = space, 29 = comma, 30 = period, 31 = question, 32 = exclamation.
char_map = {
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10,
    "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19,
    "T": 20, "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26,
    "'": 27, " ": 28, ",": 29, ".": 30, "?": 31, "!": 32
}
idx_map = {idx: ch for ch, idx in char_map.items()}

def text_to_sequence(text):
    """
    Convert a text string to a list of token indices based on char_map.
    Unrecognized characters are skipped.
    """
    seq = []
    for ch in text:
        if ch in char_map:
            seq.append(char_map[ch])
    return seq

def sequence_to_text(seq):
    """
    Convert a list of token indices to the corresponding text string.
    """
    chars = []
    for idx in seq:
        if idx in idx_map:
            chars.append(idx_map[idx])
    return "".join(chars)
