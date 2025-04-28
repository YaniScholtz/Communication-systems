import cmath as cm
import numpy as np
import math as mt
from itertools import combinations
from typing import List, Tuple, Union
import itertools as it

#########################################################################
#                  Type definitions and constants                       #
#########################################################################

BIT_SEQUENCE_TYPE = List[int]
SYMBOL_SEQUENCE_TYPE = List[complex]
BIT_TO_SYMBOL_MAP_TYPE = List[List[Union[complex, List[int]]]]
GENERATOR_MATRIX_TYPE = List[List[int]]
RANDOM_VALUES_SYMBOLS_TYPE = List[List[float]]
RANDOM_VALUES_RUN_TYPE = List[List[List[float]]]
SER_TYPE = List[float]
BER_TYPE = List[float]

MAP_BPSK: BIT_TO_SYMBOL_MAP_TYPE = [
    [(-1 + 0j), [0]],
    [(1 + 0j), [1]],
]

MAP_4QAM: BIT_TO_SYMBOL_MAP_TYPE = [
    [(1 + 1j) / cm.sqrt(2), [0, 0]],
    [(-1 + 1j) / cm.sqrt(2), [0, 1]],
    [(-1 - 1j) / cm.sqrt(2), [1, 1]],
    [(1 - 1j) / cm.sqrt(2), [1, 0]],
]

MAP_8PSK: BIT_TO_SYMBOL_MAP_TYPE = [
    [cm.rect(1, 0 * cm.pi / 4), [1, 1, 1]],
    [cm.rect(1, 1 * cm.pi / 4), [1, 1, 0]],
    [cm.rect(1, 2 * cm.pi / 4), [0, 1, 0]],
    [cm.rect(1, 3 * cm.pi / 4), [0, 1, 1]],
    [cm.rect(1, 4 * cm.pi / 4), [0, 0, 1]],
    [cm.rect(1, 5 * cm.pi / 4), [0, 0, 0]],
    [cm.rect(1, 6 * cm.pi / 4), [1, 0, 0]],
    [cm.rect(1, 7 * cm.pi / 4), [1, 0, 1]],
]

#########################################################################
#                  PLAYGROUND: Test your code here                      #
#########################################################################


def evaluate():

    return


#########################################################################
#                  OPTIONAL: WILL NOT BE EVALUATED                      #
#########################################################################


def bit_to_symbol(
    bit_sequence: BIT_SEQUENCE_TYPE, bit_to_symbol_map: BIT_TO_SYMBOL_MAP_TYPE
) -> SYMBOL_SEQUENCE_TYPE:
    """
    Converts a sequence of bits to a sequence of symbols using the bit to symbol map.

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
                             the maps that will be used are given as MAP_BPSK, MAP_4QAM and MAP_8PSK
          Example:
            [
                [ 1+1j,    [0, 0] ],
                [ -1+1j,   [0, 1] ],
                [ -1-1j,   [1, 1] ],
                [ 1-1j,    [1, 0] ]
            ]
    returns:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
    """

    length_of_sequence = len(bit_to_symbol_map[0][1])
    symbol_sequence = []

    for i in range(0, len(bit_sequence), length_of_sequence):
        bit_chunk = bit_sequence[i : i + length_of_sequence]
        for j in bit_to_symbol_map:
            if bit_chunk == j[1]:
                symbol_sequence.append(j[0])
                break

    return symbol_sequence


def symbol_to_bit(
    symbol_sequence, bit_to_symbol_map: BIT_TO_SYMBOL_MAP_TYPE
) -> BIT_SEQUENCE_TYPE:
    """
    Returns a sequence of bits that corresponds to the provided sequence of symbols containing noise using the bit to symbol map that respresent the modulation scheme and the euclidean distance

    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j]
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
                             the maps that will be used are given as MAP_BPSK, MAP_4QAM and MAP_8PSK
          Example:
            [
                [ 1+1j,    [0, 0] ],
                [ -1+1j,   [0, 1] ],
                [ -1-1j,   [1, 1] ],
                [ 1-1j,    [1, 0] ]
            ]

    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]

    """

    bit_sequence = []
    bit_seq = []
    for i in symbol_sequence:
        closest_distance = 10000
        bits = None
        for symbol, bit in bit_to_symbol_map:
            distance = abs(symbol - i) ** 2
            if distance < closest_distance:
                closest_distance = distance
                bits = bit

        bit_seq.append(bits)

    for i in bit_seq:
        for j in i:
            bit_sequence.append(j)

    return bit_sequence


#########################################################################
#                   Question 1: Linear Block Codes                      #
#########################################################################


def linear_block_codes_encode(
    bit_sequence: BIT_SEQUENCE_TYPE, generator_matrix: GENERATOR_MATRIX_TYPE
) -> BIT_SEQUENCE_TYPE:
    """
    Takes the given sequence of bits and encodes it using linear block coding and the generator matrix. The function returns the encoded sequence.

    The sequence of bits will match the size of the generator matrix.

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        generator_matrix -> type <class 'list'> : A list containing lists, making a 2D array representing a matrix. The first index refers to the row and the second index refers to the column.
          Example (example 2 from lecture notes ):
            G = [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            ]

    returns:
        codeword_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    codeword_sequence = []

    codeword_sequence = (np.dot(bit_sequence, generator_matrix) % 2).tolist()

    return codeword_sequence


def linear_block_codes_decode(
    codeword_sequence: BIT_SEQUENCE_TYPE, generator_matrix: GENERATOR_MATRIX_TYPE
) -> BIT_SEQUENCE_TYPE:
    """
    Takes the given sequence of bits (which may contain errors) and decodes it using linear block coding and the generator matrix, performing error correction coding.
    The function returns the decoded sequence. If the method can not find and correct the errors, then return the codeword_sequence.

    The sequence of bits will match the size of the generator matrix.

    parameters:
        codeword_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        generator_matrix -> type <class 'list'> : A list containing lists, making a 2D array representing a matrix. The first index refers to the row and the second index refers to the column.
          Example (example 2 from lecture notes ):
            G = [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            ]

    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    for i in range(0, len(generator_matrix)):
        if generator_matrix[-1][i] == 1:
            column = i
            break

    P = [row[column + 1 :] for row in generator_matrix]
    all_zeroes = True
    PT = np.transpose(P)
    I = np.identity(len(PT))

    H = np.concatenate((PT, I), 1)
    HT = np.transpose(H)

    z = np.matmul(codeword_sequence, HT) % 2
    HT = np.array(HT)

    z = np.array(z)
    row_single = 0
    rows = len(generator_matrix)

    for i in z:
        if i != 0:
            all_zeroes = False
            break

    if all_zeroes:
        return codeword_sequence[:rows]

    seq = []
    equal = False
    code_copy = codeword_sequence[:]

    rows_gotten = False

    rows_multiple = []
    for i in range(1, 3):
        for mix in it.combinations(HT, i):
            combined = np.sum(mix, axis=0) % 2
            if np.array_equal(combined, z):
                rows_multiple = [
                    np.where(np.all(HT == row, axis=1))[0][0] for row in mix
                ]
                rows_gotten = True
                break
        if rows_gotten:
            break

    if rows_gotten and len(rows_multiple) == 1:
        row_single = rows_multiple[0]
        code_copy[row_single] ^= 1
        seq.extend(code_copy[:])

    elif rows_gotten and len(rows_multiple) > 1:
        for row in rows_multiple:
            code_copy[row] ^= 1
        seq.extend(code_copy[:])

    else:

        return code_copy[:rows]

    return seq[:rows]


def linear_block_codes_encode_long_sequence(
    bit_sequence: BIT_SEQUENCE_TYPE, generator_matrix: GENERATOR_MATRIX_TYPE
) -> BIT_SEQUENCE_TYPE:
    """
    Takes the given sequence of bits and encodes it using linear block coding and the generator matrix. The function returns the encoded sequence.

    The length of the bit_sequence is not going to match the generator matrix length, thus the bit sequence needs to be divided into smaller sequences, encoded using
    linear_block_codes_encode, and combined into a single larger bit sequence.

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        generator_matrix -> type <class 'list'> : A list containing lists, making a 2D array representing a matrix. The first index refers to the row and the second index refers to the column.
          Example (example 2 from lecture notes ):
            G = [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            ]

    returns:
        codeword_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    num_rows = len(generator_matrix)
    codeword_sequence = []

    for i in range(0, len(bit_sequence), num_rows):
        encoded = []
        chunk = bit_sequence[i : i + num_rows]
        encoded = linear_block_codes_encode(chunk, generator_matrix)
        codeword_sequence.extend(encoded)

    return codeword_sequence


def linear_block_codes_decode_long_sequence(
    codeword_sequence: BIT_SEQUENCE_TYPE, generator_matrix: GENERATOR_MATRIX_TYPE
) -> BIT_SEQUENCE_TYPE:
    """
    Takes the given sequence of bits (which may contain errors) and decodes it using linear block coding and the generator matrix, performing error correction coding.
    The function returns the decoded sequence.

    The sequence of bits will not match the size of the generator matrix, it should thus be broken up into smaller sequences, decoded using linear_block_codes_decoding
    and combined to form a single decoded bit sequence.

    parameters:
        codeword_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        generator_matrix -> type <class 'list'> : A list containing lists, making a 2D array representing a matrix. The first index refers to the row and the second index refers to the column.
          Example (example 2 from lecture notes ):
            G = [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            ]

    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    num_columns = len(generator_matrix[0])
    bit_sequence = []
    for i in range(0, len(codeword_sequence), num_columns):
        encoded = []
        chunk = codeword_sequence[i : i + num_columns]

        encoded = linear_block_codes_decode(chunk, generator_matrix)
        bit_sequence.extend(encoded)

    return bit_sequence


#########################################################################
#                   Question 2: Convolutional Codes                     #
#########################################################################


def convolutional_codes_encode(bit_sequence: BIT_SEQUENCE_TYPE) -> BIT_SEQUENCE_TYPE:
    """
    Takes the given sequence of bits and encodes it using convolutional codes. The function returns the encoded sequence.
    The parameters for the encoder are provided in the practical guide.
    The sequence of bits can be any length.

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]

    returns:


        codeword_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    codeword_sequence = []

    state_table = {
        ("00", "0"): ("00", "000"),
        ("00", "1"): ("10", "111"),
        ("01", "0"): ("00", "001"),
        ("01", "1"): ("10", "110"),
        ("10", "0"): ("01", "010"),
        ("10", "1"): ("11", "101"),
        ("11", "0"): ("01", "011"),
        ("11", "1"): ("11", "100"),
    }

    current_state = "00"

    for bit in bit_sequence:
        bit_str = str(bit)

        next_state, output_bits = state_table[(current_state, bit_str)]

        codeword_sequence.extend([int(b) for b in output_bits])

        current_state = next_state

    return codeword_sequence


def manhattan_distance(x, y):
    distance = 0
    for bit1, bit2 in zip(x, y):
        distance += abs(int(bit1) - int(bit2))
    return distance


def convolutional_codes_decode(
    codeword_sequence: BIT_SEQUENCE_TYPE,
) -> BIT_SEQUENCE_TYPE:
    """
    Takes the given sequence of bits (which may contain errors) and decodes it using convolutional codes, performing error correction coding.
    The parameters for the encoder are provided in the practical guide.
    The sequence of bits can be any length.

    NOTE - Assume that zeros was appended to the original sequence before it was encoded and passed to this function.

    parameters:
        codeword_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]

    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    state_table = {
        "00": {"0": ("00", "000"), "1": ("10", "111")},
        "01": {"0": ("00", "001"), "1": ("10", "110")},
        "10": {"0": ("01", "010"), "1": ("11", "101")},
        "11": {"0": ("01", "011"), "1": ("11", "100")},
    }

    states = ["00", "01", "10", "11"]

    num_bits = len(codeword_sequence) // 3
    num_states = len(states)
    path = []
    for _ in range(num_bits + 1):
        path.append([None] * num_states)

    initial_state_index = states.index("00")

    trellis = np.full((num_bits + 1, num_states), np.inf)

    trellis[0][initial_state_index] = 0

    for i in range(num_bits):
        start_index = i * 3
        end_index = start_index + 3

        received_bits = codeword_sequence[start_index:end_index]
        for state in states:
            for input_bit in ["0", "1"]:
                next_state, output_bits = state_table[state][input_bit]

                next_state_index = states.index(next_state)
                state_index = states.index(state)

                output_bits_list = []
                for b in output_bits:
                    output_bits_list.append(int(b))

                metric = manhattan_distance(received_bits, output_bits_list)

                total_metric = trellis[i][state_index] + metric

                if total_metric < trellis[i + 1][next_state_index]:
                    trellis[i + 1][next_state_index] = total_metric
                    path[i + 1][next_state_index] = (
                        state_index,
                        input_bit,
                    )

    decoded_bits = []
    current_state_index = states.index("00")
    for i in range(num_bits, 0, -1):
        prev_state_index, bit = path[i][current_state_index]

        decoded_bits.append(int(bit))
        current_state_index = prev_state_index

    decoded_bits.reverse()
    decoded_bits = decoded_bits[:-2]
    return decoded_bits


def convolutional_codes_encode_long_sequence(
    bit_sequence: BIT_SEQUENCE_TYPE,
) -> BIT_SEQUENCE_TYPE:
    """
    Takes the given sequence of bits and encodes it using convolutional codes. The function returns the encoded sequence.
    The parameters for the encoder are provided in the practical guide.

    The sequence of bits should be broken up into smaller sequences, zeros should be appended to the end of each sequence, and then encoded using convolutional_codes_encode
    to yield multiple sequences of length Nc = 300. All the encoded sequences should then be combined to form a single larger sequence.

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        generator_matrix -> type <class 'list'> : A list containing lists, making a 2D array representing a matrix. The first index refers to the row and the second index refers to the column.
          Example (example 2 from lecture notes ):
            G = [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            ]

    returns:
        codeword_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    chunk_size = 98
    codeword_sequence = []

    for i in range(0, len(bit_sequence), chunk_size):
        chunk = bit_sequence[i : i + chunk_size]
        chunk = chunk + [0, 0]
        encoded_chunk = convolutional_codes_encode(chunk)
        codeword_sequence.extend(encoded_chunk)

    return codeword_sequence


def convolutional_codes_decode_long_sequence(
    codeword_sequence: BIT_SEQUENCE_TYPE,
) -> BIT_SEQUENCE_TYPE:
    """
    Takes the given sequence of bits (which may contain errors) and decodes it using convolutional codes, performing error correction coding.
    The parameters for the encoder are provided in the practical guide.

    The sequence will consist of multiple codewords sequences of length 300, which should be decoded using convolutional_codes_decode,
    and then recombined into a single decoded sequence with the appended zeros removed.

    parameters:
        codeword_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]

    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    chunk_size = 300
    bit_sequence = []

    for i in range(0, len(codeword_sequence), chunk_size):
        chunk = codeword_sequence[i : i + chunk_size]
        decoded_chunk = convolutional_codes_decode(chunk)
        bit_sequence.extend(decoded_chunk)

    return bit_sequence


#########################################################################
#                   Question 3: AWGN and BER                            #
#########################################################################


def AWGN_Transmission(
    bit_sequence: BIT_SEQUENCE_TYPE,
    bit_to_symbol_map: BIT_TO_SYMBOL_MAP_TYPE,
    random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE,
    snr: float,
    transmission_rate: float,
) -> Tuple[SYMBOL_SEQUENCE_TYPE, BIT_SEQUENCE_TYPE]:
    """
    This function takes the given bit sequence, modulate it using the bit to symbol map, add noise to the symbol sequence as described in the practical guide,
    and demodulate it back into a noisy bit sequence. The function returns this generated noisy bit sequence, along with the noisy symbol sequence with the added noise.

    NOTE - As with the previous practicals, BPSK uses a different equation

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
                             the maps that will be used are given as MAP_BPSK, MAP_4QAM and MAP_8PSK
          Example:
            [
                [ 1+1j,    [0, 0] ],
                [ -1+1j,   [0, 1] ],
                [ -1-1j,   [1, 1] ],
                [ 1-1j,    [1, 0] ]
            ]
        random_values_for_symbols -> type <class 'list'> : List containing lists. Each entry is a list containing two values, which are random Gaussian zero mean unity variance values.
                                                           The first index refers to the symbol in the sequence, and the second index refers to the real or imaginary kappa value.
          Example:
            [[1.24, 0.42], [-1.2, -0.3], [0, 1.23], [-0.3, 1.2]]
        snr -> type <class 'float'> : A float value which is the SNR that should be used when adding noise
        transmission_rate -> type <class 'flaot'> : A float value which is the transmission rate (Rc), which should be used when adding noise.

    returns:
        noisy_symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j]
        noisy_bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """
    count = len(bit_to_symbol_map)
    symbol_sequence = bit_to_symbol(bit_sequence, bit_to_symbol_map)
    noisy_symbol_sequence = []
    fbit = len(bit_to_symbol_map[0][1])
    sigma = 1 / np.sqrt(10 ** (snr / 10) * transmission_rate * fbit)
    for i in range(len(symbol_sequence)):
        noisy_symbol = None
        if count == 2:
            noisy_symbol = symbol_sequence[i] + sigma * random_values_for_symbols[i][0]
            noisy_symbol_sequence.append(noisy_symbol)

        else:
            noise = complex(
                random_values_for_symbols[i][0], random_values_for_symbols[i][1]
            ) / cm.sqrt(2)
            noisy_symbol = symbol_sequence[i] + sigma * noise
            noisy_symbol_sequence.append(noisy_symbol)

    noisy_bit_sequence = []

    noisy_bit_sequence = symbol_to_bit(noisy_symbol_sequence, bit_to_symbol_map)

    return noisy_symbol_sequence, noisy_bit_sequence


def BER_linear_block_codes(
    bit_sequence: BIT_SEQUENCE_TYPE,
    generator_matrix: GENERATOR_MATRIX_TYPE,
    bit_to_symbol_map: BIT_TO_SYMBOL_MAP_TYPE,
    random_values_for_runs: RANDOM_VALUES_RUN_TYPE,
    snr_range: List[float],
) -> List[float]:
    """
    This functions simulates the linear block codes method over a AWGN channel for different snr values. The snr_range argument, provides the snr values that should be used for
    each run. Each run encodes the long bit sequence using linear block codes, transmits it over the AWGN channel, decodes it, and then calculate the BER using the input bit sequence
    and the final bit sequence. This is repeated for each snr value, and the results for each run is stored in a list and returned.

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
                             the maps that will be used are given as MAP_BPSK, MAP_4QAM and MAP_8PSK
          Example:
            [
                [ 1+1j,    [0, 0] ],
                [ -1+1j,   [0, 1] ],
                [ -1-1j,   [1, 1] ],
                [ 1-1j,    [1, 0] ]
            ]
        random_values_for_symbols -> type <class 'list'> : List containing lists containing lists. Each entry is a list containing multiple lists with two values, which are random Gaussian zero mean unity variance values.
                                                           The first index refers to run (corresponding with the snr value),
                                                           The second index refers to the symbol within that run,
                                                           The third index refers to the real or imaginary kappa value
          Example:
            [
                [1.24, 0.42], [-1.2, -0.3], [0, 1.23], [-0.3, 1.2],
                [-0.3, 0.42], [-0.32, 0.42], [1.24, 1.23], [-0.3, 1.2],
                [1.24, 1.24], [-1.2, 0.42], [0, 1.23], [0.42, -0.3],
            ]
        snr_range -> type <class 'list'> : A list containing float values which are the SNR that should be used when adding noise during the different runs

    returns:
        BER_values -> type <class 'list'> : A list containing float values which are the BER results for the different runs, corresponding to the snr value in the snr_range.
    """

    BER_values = []
    for i in range(len(snr_range)):
        encoding = []
        noise = 0
        decoding = []
        BER = None
        BER_log = None
        bit_errors = 0
        encoding = linear_block_codes_encode_long_sequence(
            bit_sequence, generator_matrix
        )

        noise_symbols, noisy_bits = AWGN_Transmission(
            encoding, bit_to_symbol_map, random_values_for_runs[i], snr_range[i], 0.5
        )

        decoding = linear_block_codes_decode_long_sequence(noisy_bits, generator_matrix)

        for j in range(len(decoding)):
            if bit_sequence[j] != decoding[j]:
                bit_errors += 1

        BER = bit_errors / len(bit_sequence)
        BER_log = None if BER == 0 else np.real(cm.log10(BER))

        BER_values.append(BER_log)

    return BER_values


def BER_convolution_codes(
    bit_sequence: BIT_SEQUENCE_TYPE,
    bit_to_symbol_map: BIT_TO_SYMBOL_MAP_TYPE,
    random_values_for_runs: RANDOM_VALUES_RUN_TYPE,
    snr_range: List[float],
) -> List[float]:
    """
    This functions simulates the convolutional codes method over a AWGN channel for different snr values. The snr_range argument, provides the snr values that should be used for
    each run. Each run encodes the long bit sequence using convolutional codes, transmits it over the AWGN channel, decodes it, and then calculate the BER using the input bit sequence
    and the final bit sequence. This is repeated for each snr value, and the results for each run is stored in a list and returned.

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
                             the maps that will be used are given as MAP_BPSK, MAP_4QAM and MAP_8PSK
          Example:
            [
                [ 1+1j,    [0, 0] ],
                [ -1+1j,   [0, 1] ],
                [ -1-1j,   [1, 1] ],
                [ 1-1j,    [1, 0] ]
            ]
        random_values_for_symbols -> type <class 'list'> : List containing lists containing lists. Each entry is a list containing multiple lists with two values, which are random Gaussian zero mean unity variance values.
                                                           The first index refers to run (corresponding with the snr value),
                                                           The second index refers to the symbol within that run,
                                                           The third index refers to the real or imaginary kappa value
          Example:
            [
                [1.24, 0.42], [-1.2, -0.3], [0, 1.23], [-0.3, 1.2],
                [-0.3, 0.42], [-0.32, 0.42], [1.24, 1.23], [-0.3, 1.2],
                [1.24, 1.24], [-1.2, 0.42], [0, 1.23], [0.42, -0.3],
            ]
        snr_range -> type <class 'list'> : A list containing float values which are the SNR that should be used when adding noise during the different runs

    returns:
        BER_values -> type <class 'list'> : A list containing float values which are the BER results for the different runs, corresponding to the snr value in the snr_range.
    """

    BER_values = []
    for i in range(len(snr_range)):
        encoding = []
        noise = 0
        decoding = []
        BER = None
        BER_log = None
        bit_errors = 0
        encoding = convolutional_codes_encode_long_sequence(bit_sequence)

        noise_symbols, noisy_bits = AWGN_Transmission(
            encoding, bit_to_symbol_map, random_values_for_runs[i], snr_range[i], 1 / 3
        )

        decoding = convolutional_codes_decode_long_sequence(noisy_bits)

        for j in range(len(decoding)):
            if bit_sequence[j] != decoding[j]:
                bit_errors += 1

        BER = bit_errors / len(bit_sequence)
        BER_log = None if BER == 0 else np.real(cm.log10(BER))

        BER_values.append(BER_log)

    return BER_values


######## DO NOT EDIT ########
if __name__ == "__main__":
    evaluate()
#############################
