import cmath as cm
import math
from typing import List, Union, Tuple
import numpy as np


############################# Defining Types ############################

BIT_SEQUENCE_TYPE = List[int]
SYMBOL_SEQUENCE_TYPE = List[complex]
BIT_TO_SYMBOL_MAP_TYPE = List[List[Union[complex, List[int]]]]
SYMBOL_BLOCKS_TYPE = List[List[complex]]
CHANNEL_IMPULSE_RESPONSE_TYPE = List[complex]
RANDOM_VALUES_SYMBOLS_TYPE = List[List[List[float]]]
RANDOM_VALUES_CIR_TYPE = List[List[List[float]]]
NOISY_SYMBOL_SEQUENCE_TYPE = List[List[complex]]
SER_TYPE = Union[float, None]
BER_TYPE = Union[float, None]

#########################################################################
#                   Given Modulation Bit to Symbol Maps                 #
#########################################################################

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


#########################################################################
#                           Evaluation Function                         #
#########################################################################
def evaluate():
    """
    Your code used to evaluate your system should be written here.
             !!! NOTE: This function will not be marked !!!
    """


#########################################################################
#                           Assisting Functions                         #
#########################################################################


def assist_bit_to_symbol(
    bit_sequence: BIT_SEQUENCE_TYPE, bit_to_symbol_map: BIT_TO_SYMBOL_MAP_TYPE
) -> SYMBOL_SEQUENCE_TYPE:
    """
    Converts a sequence of bits to a sequence of symbols using the bit to symbol map.

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
                             the maps that will be used are given as MAP_BPSK and MAP_4QAM
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


def assist_symbol_to_bit(
    symbol_sequence: SYMBOL_SEQUENCE_TYPE, bit_to_symbol_map: BIT_TO_SYMBOL_MAP_TYPE
) -> BIT_SEQUENCE_TYPE:
    """
    Returns a sequence of bits that corresponds to the provided sequence of symbols containing noise using the bit to symbol map that respresent the modulation scheme and the euclidean distance

    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
                             the maps that will be used are given as MAP_BPSK and MAP_4QAM
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


def assist_split_symbols_into_blocks(
    symbol_sequence: SYMBOL_SEQUENCE_TYPE, block_size: int
) -> SYMBOL_BLOCKS_TYPE:
    """
    Divides the given symbol sequence into blocks of length block_size, that the DFE and MLSE algorithm should be performed upon.

    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block

    returns:
        symbol_blocks -> type <class 'list'> : List of lists. Each list entry should be a list representing a symbol sequence, which is a list containing containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
    """

    symbol_blocks = []

    for i in range(0, len(symbol_sequence), block_size):
        bit_chunk = symbol_sequence[i : i + block_size]
        symbol_blocks.append(bit_chunk)

    return symbol_blocks


def assist_combine_blocks_into_symbols(
    symbol_blocks: SYMBOL_BLOCKS_TYPE,
) -> SYMBOL_SEQUENCE_TYPE:
    """
    Combines the given blocks of symbol sequences into a single sequence of symbols.

    parameters:
        symbol_blocks -> type <class 'list'> : List of lists. Each list entry should be a list representing a symbol sequence, which is a list containing containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]

    returns:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]

    """

    symbol_sequence = []

    for i in symbol_blocks:
        for j in i:
            symbol_sequence.append(j)

    return symbol_sequence


#########################################################################
#                         DFE and MLSE Functions                        #
#########################################################################


def DFE_BPSK_BLOCK(
    symbol_sequence: SYMBOL_SEQUENCE_TYPE,
    impulse_response: CHANNEL_IMPULSE_RESPONSE_TYPE,
) -> SYMBOL_SEQUENCE_TYPE:
    """
    Performs the DFE algorithm on the given symbol sequence (which was modulated using the BPSK scheme) with the given impulse response, and returns the most probable transmitted symbol sequence.
    The impulse response length can be assumed to be 3, and the prepended symbols for t=-1 and t=-2 can be assumed to be [1, 1]
    Only the transmitted data bits must be returned, thus exluding the prepended symbols. Thus len(symbol_sequence) equals len(transmitted_sequence).

    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        impulse_response -> type <class 'list'> : List containing complex items (<class "complex">) which represents the impulse response coeficients for example [1+1j, 2+2j, -0.66-0.25j] represents [c0, c1, c2]

    returns:
        transmitted_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]

    """
    s = [1, 1]
    count = 2

    for t in range(0, len(symbol_sequence)):
        delta_1 = (
            abs(
                symbol_sequence[t]
                - (
                    impulse_response[0] * 1
                    + impulse_response[1] * s[count - 1]
                    + impulse_response[2] * s[count - 2]
                )
            )
            ** 2
        )
        delta_0 = (
            abs(
                symbol_sequence[t]
                - (
                    impulse_response[0] * -1
                    + impulse_response[1] * s[count - 1]
                    + impulse_response[2] * s[count - 2]
                )
            )
            ** 2
        )

        min_valu = min(delta_1, delta_0)
        if min_valu == delta_1:
            s.append(1)
        else:
            s.append(-1)

        count += 1

    transmitted_sequence = s[2:]

    return transmitted_sequence


def DFE_4QAM_BLOCK(
    symbol_sequence: SYMBOL_SEQUENCE_TYPE,
    impulse_response: CHANNEL_IMPULSE_RESPONSE_TYPE,
) -> SYMBOL_SEQUENCE_TYPE:
    """
    Performs the DFE algorithm on the given symbol sequence (which was modulated using the 4QAM scheme) with the given impulse response, and returns the most probable transmitted symbol sequence.
    The impulse response length can be assumed to be 3, and the prepended symbols for t=-1 and t=-2 can be assumed to be [(0.7071067811865475+0.7071067811865475j), (0.7071067811865475+0.7071067811865475j)]
    Only the transmitted data bits must be returned, thus exluding the prepended symbols. Thus len(symbol_sequence) equals len(transmitted_sequence).

    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        impulse_response -> type <class 'list'> : List containing complex items (<class "complex">) which represents the impulse response coeficients for example [1+1j, 2+2j, -0.66-0.25j] represents [c0, c1, c2]

    returns:
        transmitted_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]

    """

    s = [
        0.7071067811865475 + 0.7071067811865475j,
        0.7071067811865475 + 0.7071067811865475j,
    ]
    QAM_counter = 2

    for t in range(0, len(symbol_sequence)):
        delta_1 = (
            abs(
                symbol_sequence[t]
                - (
                    impulse_response[0] * (1 + 1j) / cm.sqrt(2)
                    + impulse_response[1] * s[QAM_counter - 1]
                    + impulse_response[2] * s[QAM_counter - 2]
                )
            )
            ** 2
        )
        delta_j = (
            abs(
                symbol_sequence[t]
                - (
                    impulse_response[0] * (-1 + 1j) / cm.sqrt(2)
                    + impulse_response[1] * s[QAM_counter - 1]
                    + impulse_response[2] * s[QAM_counter - 2]
                )
            )
            ** 2
        )

        delta_Minus1 = (
            abs(
                symbol_sequence[t]
                - (
                    impulse_response[0] * (1 - 1j) / cm.sqrt(2)
                    + impulse_response[1] * s[QAM_counter - 1]
                    + impulse_response[2] * s[QAM_counter - 2]
                )
            )
            ** 2
        )

        delta_minusj = (
            abs(
                symbol_sequence[t]
                - (
                    impulse_response[0] * (-1 - 1j) / cm.sqrt(2)
                    + impulse_response[1] * s[QAM_counter - 1]
                    + impulse_response[2] * s[QAM_counter - 2]
                )
            )
            ** 2
        )

        min_val = min(delta_1, delta_j, delta_Minus1, delta_minusj)

        if min_val == delta_1:
            s.append((1 + 1j) / cm.sqrt(2))
        elif delta_j == min_val:
            s.append((-1 + 1j) / cm.sqrt(2))
        elif delta_Minus1 == min_val:
            s.append((1 - 1j) / cm.sqrt(2))
        elif delta_minusj == min_val:
            s.append((-1 - 1j) / cm.sqrt(2))

        QAM_counter = QAM_counter + 1

    transmitted_sequence = s[2:]

    return transmitted_sequence


def MLSE_BPSK_BLOCK(
    symbol_sequence: SYMBOL_SEQUENCE_TYPE,
    impulse_response: CHANNEL_IMPULSE_RESPONSE_TYPE,
) -> SYMBOL_SEQUENCE_TYPE:
    """
    Performs the MLSE algorithm on the given symbol sequence (which was modulated using the BPSK scheme) with the given impulse response, and returns the most probable transmitted symbol sequence.
    The impulse response length can be assumed to be 3, and the prepended symbols for t=-1 and t=-2 can be assumed to be [1, 1]

    !!! NOTE: The appended symbols should be included in the given symbol sequence, thus if the block size is 200, then the length of the given symbol sequence should be 202.

    Only the transmitted data bits must be returned, thus exluding the prepended symbols AND the appended symbols. Thus is the block size is 200 then len(transmitted_sequence) should be 200.

    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        impulse_response -> type <class 'list'> : List containing complex items (<class "complex">) which represents the impulse response coeficients for example [1+1j, 2+2j, -0.66-0.25j] represents [c0, c1, c2]

    returns:
        transmitted_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
    """
    transmitted_sequence = []
    return transmitted_sequence


def MLSE_4QAM_BLOCK(symbol_sequence: list, impulse_response: list) -> list:
    """
    Performs the MLSE algorithm on the given symbol sequence (which was modulated using the 4QAM scheme) with the given impulse response, and returns the most probable transmitted symbol sequence.
    The impulse response length can be assumed to be 3, and the prepended symbols for t=-1 and t=-2 can be assumed to be [(0.7071067811865475+0.7071067811865475j), (0.7071067811865475+0.7071067811865475j)]

    !!! NOTE: The appended symbols should be included in the given symbol sequence, thus if the block size is 200, then the length of the given symbol sequence should be 202.

    Only the transmitted data bits must be returned, thus exluding the prepended symbols AND the appended symbols. Thus is the block size is 200 then len(transmitted_sequence) should be 200.

    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        impulse_response -> type <class 'list'> : List containing complex items (<class "complex">) which represents the impulse response coeficients for example [1+1j, 2+2j, -0.66-0.25j] represents [c0, c1, c2]

    returns:
        transmitted_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]

    """
    transmitted_sequence = []
    return transmitted_sequence


#########################################################################
#                         SER and BER Functions                         #
#########################################################################

# BPSK


def SER_BER_BPSK_DFE_STATIC(
    bit_sequence: BIT_SEQUENCE_TYPE,
    block_size: int,
    random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE,
    snr: float,
) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using BPSK
        - splits the symbol sequence into blocks with the given block size
        - add noise to the symbol sequence in each block using the equation in the practical guide and the static impulse response and SNR
        - save the noisy symbol sequence from each block
        - perform the DFE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            !!! NOTE: The imaginary component in this case will be 0 since for BPSK no imaginary noise will be added and the other equation shown in the pre-practical slides should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.

    """
    CIR = [0.29 + 0.98j, 0.73 - 0.24j, 0.21 + 0.91j]
    symbol_sequence = assist_bit_to_symbol(bit_sequence, MAP_BPSK)
    transmitted_sequence = []
    blocks = assist_split_symbols_into_blocks(symbol_sequence, block_size)
    noisy_symbol_blocks = []

    sigma = 1 / cm.sqrt(10 ** (snr / 10) * 1)
    SERerrors = 0
    BERerrors = 0

    for block_index in range(len(blocks)):
        noisy_symbols = []
        blocks[block_index] = [1, 1] + blocks[block_index]
        for symbol_index in range(2, len(blocks[block_index])):

            noise_values = random_values_for_symbols[block_index][symbol_index - 2][0]

            noise_val = sigma * noise_values

            rt = (
                blocks[block_index][symbol_index] * CIR[0]
                + blocks[block_index][symbol_index - 1] * CIR[1]
                + blocks[block_index][symbol_index - 2] * CIR[2]
                + noise_val
            )

            noisy_symbols.append(rt)
        noisy_symbol_blocks.append(noisy_symbols)

    for i in range(len(noisy_symbol_blocks)):

        transmitted = DFE_BPSK_BLOCK(noisy_symbol_blocks[i], CIR)

        transmitted_sequence.append(transmitted)

    transitted_sequence_combined = assist_combine_blocks_into_symbols(
        transmitted_sequence
    )

    for i in range(len(symbol_sequence)):
        if symbol_sequence[i] != transitted_sequence_combined[i]:
            SERerrors += 1

    SER_begin = SERerrors / len(symbol_sequence)
    SER_log = None if SER_begin == 0 else math.log10(SER_begin)
    SER = SER_log
    bit_sequence_combined = assist_symbol_to_bit(transitted_sequence_combined, MAP_BPSK)

    for i in range(len(bit_sequence)):
        if bit_sequence[i] != bit_sequence_combined[i]:
            BERerrors += 1

    BER_begin = BERerrors / len(bit_sequence)
    BER_log = None if BER_begin == 0 else math.log10(BER_begin)
    BER = BER_log

    return noisy_symbol_blocks, SER, BER


def SER_BER_BPSK_DFE_DYNAMIC(
    bit_sequence: BIT_SEQUENCE_TYPE,
    block_size: int,
    random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE,
    random_values_for_CIR: RANDOM_VALUES_CIR_TYPE,
    snr: float,
) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using BPSK
        - splits the symbol sequence into blocks with the given block size
        - calculate a channel impulse response for each block using the dynamic impulse response equation in the practical guide
        - add noise to the symbol sequence in each block using the equation in the practical guide and the dynamic impulse response and SNR
        - save the noisy symbol sequence from each block
        - perform the DFE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            !!! NOTE: The imaginary component in this case will be 0 since for BPSK no imaginary noise will be added and the other equation shown in the pre-practical slides should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        random_values_for_CIR -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_CIR[0][0][0] and returns a float.
            These random values are used to calculate the dynamic impulse response using equation 4 in the practical guide
            The first index indicates the corresponding block. The second index indicate the corresponding coeficient (c0 - [0], c1 = [1], c2 - [2]). The third index indicates the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_CIR[7][2][1] -> is the float that should be used for the imaginary component kappa value to calculate the c2 coeficient for the 8th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.

    """
    symbol_sequence = assist_bit_to_symbol(bit_sequence, MAP_BPSK)
    transmitted_sequence = []
    blocks = assist_split_symbols_into_blocks(symbol_sequence, block_size)
    noisy_symbol_blocks = []

    sigma = 1 / cm.sqrt(10 ** (snr / 10) * 1)
    SERerrors = 0
    BERerrors = 0
    cdynamic = []

    for block_index in range(len(blocks)):
        noisy_symbols = []
        blocks[block_index] = [
            (1),
            (1),
        ] + blocks[block_index]
        for symbol_index in range(2, len(blocks[block_index])):

            cdynamic = [
                (
                    random_values_for_CIR[block_index][0][0]
                    + 1j * random_values_for_CIR[block_index][0][1]
                )
                / cm.sqrt(6),
                (
                    random_values_for_CIR[block_index][1][0]
                    + 1j * random_values_for_CIR[block_index][1][1]
                )
                / cm.sqrt(6),
                (
                    random_values_for_CIR[block_index][2][0]
                    + 1j * random_values_for_CIR[block_index][2][1]
                )
                / cm.sqrt(6),
            ]
            noise_values = random_values_for_symbols[block_index][symbol_index - 2][0]

            noise_val = sigma * noise_values

            rt = (
                blocks[block_index][symbol_index] * cdynamic[0]
                + blocks[block_index][symbol_index - 1] * cdynamic[1]
                + blocks[block_index][symbol_index - 2] * cdynamic[2]
                + noise_val
            )

            noisy_symbols.append(rt)
        noisy_symbol_blocks.append(noisy_symbols)

    for i in range(len(noisy_symbol_blocks)):
        cdynamic = [
            (
                random_values_for_CIR[block_index][0][0]
                + 1j * random_values_for_CIR[block_index][0][1]
            )
            / cm.sqrt(6),
            (
                random_values_for_CIR[block_index][1][0]
                + 1j * random_values_for_CIR[block_index][1][1]
            )
            / cm.sqrt(6),
            (
                random_values_for_CIR[block_index][2][0]
                + 1j * random_values_for_CIR[block_index][2][1]
            )
            / cm.sqrt(6),
        ]

        transmitted = DFE_BPSK_BLOCK(noisy_symbol_blocks[i], cdynamic)

        transmitted_sequence.append(transmitted)

    transitted_sequence_combined = assist_combine_blocks_into_symbols(
        transmitted_sequence
    )

    for i in range(len(symbol_sequence)):
        if symbol_sequence[i] != transitted_sequence_combined[i]:
            SERerrors += 1

    SER_begin = SERerrors / len(symbol_sequence)
    SER_log = None if SER_begin == 0 else math.log10(SER_begin)
    SER = SER_log
    bit_sequence_combined = assist_symbol_to_bit(transitted_sequence_combined, MAP_BPSK)

    for i in range(len(bit_sequence)):
        if bit_sequence[i] != bit_sequence_combined[i]:
            BERerrors += 1

    BER_begin = BERerrors / len(bit_sequence)
    BER_log = None if BER_begin == 0 else math.log10(BER_begin)
    BER = BER_log

    return noisy_symbol_blocks, SER, BER


def SER_BER_BPSK_MLSE_STATIC(
    bit_sequence: BIT_SEQUENCE_TYPE,
    block_size: int,
    random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE,
    snr: float,
) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using BPSK
        - splits the symbol sequence into blocks with the given block size
        - add the appended symbols to each sequence in each block. For BPSK the appended symbols are [1, 1]
        - add noise to the symbol sequence including the appended symbols in each block using the equation in the practical guide and the static impulse response and SNR.
        - save the noisy symbol sequence from each block which should all be block size + 2 long, thus including the symbols that was appended.
        - perform the MLSE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences excluding the appended symbols.
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            !!! NOTE: The imaginary component in this case will be 0 since for BPSK no imaginary noise will be added and the other equation shown in the pre-practical slides should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.

    """
    CIR = [0.29 + 0.98j, 0.73 - 0.24j, 0.21 + 0.91j]
    symbol_sequence = assist_bit_to_symbol(bit_sequence, MAP_BPSK)
    transmitted_sequence = []
    blocks = assist_split_symbols_into_blocks(symbol_sequence, block_size)
    noisy_symbol_blocks = []

    sigma = 1 / cm.sqrt(10 ** (snr / 10) * cm.log(2, 2))
    SERerrors = 0
    BERerrors = 0

    for block_index in range(len(blocks)):
        noisy_symbols = []
        blocks[block_index] = [1, 1] + blocks[block_index] + [1, 1]
        for symbol_index in range(2, len(blocks[block_index])):

            noise_values = random_values_for_symbols[block_index][symbol_index - 2][0]

            noise_val = sigma * noise_values

            rt = (
                blocks[block_index][symbol_index] * CIR[0]
                + blocks[block_index][symbol_index - 1] * CIR[1]
                + blocks[block_index][symbol_index - 2] * CIR[2]
                + noise_val
            )

            noisy_symbols.append(rt)
        noisy_symbol_blocks.append(noisy_symbols)

    for i in range(len(noisy_symbol_blocks)):

        transmitted = MLSE_BPSK_BLOCK(noisy_symbol_blocks[i], CIR)

        transmitted_sequence.append(transmitted)

    transitted_sequence_combined = assist_combine_blocks_into_symbols(
        transmitted_sequence
    )

    for i in range(len(symbol_sequence)):
        if symbol_sequence[i] != transitted_sequence_combined[i]:
            SERerrors += 1

    SER_begin = SERerrors / len(symbol_sequence)
    SER_log = None if SER_begin == 0 else math.log10(SER_begin)
    SER = SER_log
    bit_sequence_combined = assist_symbol_to_bit(transitted_sequence_combined, MAP_BPSK)

    for i in range(len(bit_sequence)):
        if bit_sequence[i] != bit_sequence_combined[i]:
            BERerrors += 1

    BER_begin = BERerrors / len(bit_sequence)
    BER_log = None if BER_begin == 0 else math.log10(BER_begin)
    BER = BER_log

    return noisy_symbol_blocks, SER, BER


def SER_BER_BPSK_MLSE_DYNAMIC(
    bit_sequence: BIT_SEQUENCE_TYPE,
    block_size: int,
    random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE,
    random_values_for_CIR: RANDOM_VALUES_CIR_TYPE,
    snr: float,
) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using BPSK
        - splits the symbol sequence into blocks with the given block size
        - add the appended symbols to each sequence in each block. For BPSK the appended symbols are [1, 1]
        - calculate a channel impulse response for each block using the dynamic impulse response equation in the practical guide
        - add noise to the symbol sequence including the appended symbols in each block using the equation in the practical guide and the static impulse response and SNR.
        - save the noisy symbol sequence from each block which should all be block size + 2 long, thus including the symbols that was appended.
        - perform the MLSE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences excluding the appended symbols.
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            !!! NOTE: The imaginary component in this case will be 0 since for BPSK no imaginary noise will be added and the other equation shown in the pre-practical slides should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        random_values_for_CIR -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_CIR[0][0][0] and returns a float.
            These random values are used to calculate the dynamic impulse response using equation 4 in the practical guide
            The first index indicates the corresponding block. The second index indicate the corresponding coeficient (c0 - [0], c1 = [1], c2 - [2]). The third index indicates the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_CIR[7][2][1] -> is the float that should be used for the imaginary component kappa value to calculate the c2 coeficient for the 8th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.

    """

    symbol_sequence = assist_bit_to_symbol(bit_sequence, MAP_BPSK)
    transmitted_sequence = []
    blocks = assist_split_symbols_into_blocks(symbol_sequence, block_size)
    noisy_symbol_blocks = []

    sigma = 1 / cm.sqrt(10 ** (snr / 10) * 1)
    SERerrors = 0
    BERerrors = 0
    cdynamic = []

    for block_index in range(len(blocks)):
        noisy_symbols = []
        blocks[block_index] = [1, 1] + blocks[block_index] + [1, 1]
        for symbol_index in range(2, len(blocks[block_index])):

            cdynamic = [
                (
                    random_values_for_CIR[block_index][0][0]
                    + 1j * random_values_for_CIR[block_index][0][1]
                )
                / cm.sqrt(6),
                (
                    random_values_for_CIR[block_index][1][0]
                    + 1j * random_values_for_CIR[block_index][1][1]
                )
                / cm.sqrt(6),
                (
                    random_values_for_CIR[block_index][2][0]
                    + 1j * random_values_for_CIR[block_index][2][1]
                )
                / cm.sqrt(6),
            ]
            noise_values = random_values_for_symbols[block_index][symbol_index - 2][0]

            noise_val = sigma * noise_values

            rt = (
                blocks[block_index][symbol_index] * cdynamic[0]
                + blocks[block_index][symbol_index - 1] * cdynamic[1]
                + blocks[block_index][symbol_index - 2] * cdynamic[2]
                + noise_val
            )

            noisy_symbols.append(rt)
        noisy_symbol_blocks.append(noisy_symbols)

    for i in range(len(noisy_symbol_blocks)):
        cdynamic = [
            (
                random_values_for_CIR[block_index][0][0]
                + 1j * random_values_for_CIR[block_index][0][1]
            )
            / cm.sqrt(6),
            (
                random_values_for_CIR[block_index][1][0]
                + 1j * random_values_for_CIR[block_index][1][1]
            )
            / cm.sqrt(6),
            (
                random_values_for_CIR[block_index][2][0]
                + 1j * random_values_for_CIR[block_index][2][1]
            )
            / cm.sqrt(6),
        ]

        transmitted = MLSE_BPSK_BLOCK(noisy_symbol_blocks[i], cdynamic)

        transmitted_sequence.append(transmitted)

    transitted_sequence_combined = assist_combine_blocks_into_symbols(
        transmitted_sequence
    )

    for i in range(len(symbol_sequence)):
        if symbol_sequence[i] != transitted_sequence_combined[i]:
            SERerrors += 1

    SER_begin = SERerrors / len(symbol_sequence)
    SER_log = None if SER_begin == 0 else math.log10(SER_begin)
    SER = SER_log
    bit_sequence_combined = assist_symbol_to_bit(transitted_sequence_combined, MAP_BPSK)

    for i in range(len(bit_sequence)):
        if bit_sequence[i] != bit_sequence_combined[i]:
            BERerrors += 1

    BER_begin = BERerrors / len(bit_sequence)
    BER_log = None if BER_begin == 0 else math.log10(BER_begin)
    BER = BER_log

    return noisy_symbol_blocks, SER, BER


# 4QAM


def SER_BER_4QAM_DFE_STATIC(
    bit_sequence: BIT_SEQUENCE_TYPE,
    block_size: int,
    random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE,
    snr: float,
) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using 4QAM
        - splits the symbol sequence into blocks with the given block size
        - add noise to the symbol sequence in each block using the equation in the practical guide and the static impulse response and SNR
        - save the noisy symbol sequence from each block
        - perform the DFE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.

    """
    CIR = [0.29 + 0.98j, 0.73 - 0.24j, 0.21 + 0.91j]
    symbol_sequence = assist_bit_to_symbol(bit_sequence, MAP_4QAM)
    transmitted_sequence = []
    blocks = assist_split_symbols_into_blocks(symbol_sequence, block_size)
    noisy_symbol_blocks = []

    sigma = 1 / cm.sqrt(10 ** (snr / 10) * 2)
    SERerrors = 0
    BERerrors = 0

    for block_index in range(len(blocks)):
        noisy_symbols = []
        blocks[block_index] = [
            (0.7071067811865475 + 0.7071067811865475j),
            (0.7071067811865475 + 0.7071067811865475j),
        ] + blocks[block_index]
        for symbol_index in range(2, len(blocks[block_index])):

            noise_values = (
                random_values_for_symbols[block_index][symbol_index - 2][0]
                + (random_values_for_symbols[block_index][symbol_index - 2][1]) * 1j
            ) / cm.sqrt(2)

            noise_val = sigma * noise_values

            rt = (
                blocks[block_index][symbol_index] * CIR[0]
                + blocks[block_index][symbol_index - 1] * CIR[1]
                + blocks[block_index][symbol_index - 2] * CIR[2]
                + noise_val
            )

            noisy_symbols.append(rt)
        noisy_symbol_blocks.append(noisy_symbols)

    for i in range(len(noisy_symbol_blocks)):

        transmitted = DFE_4QAM_BLOCK(noisy_symbol_blocks[i], CIR)

        transmitted_sequence.append(transmitted)

    transitted_sequence_combined = assist_combine_blocks_into_symbols(
        transmitted_sequence
    )

    for i in range(len(symbol_sequence)):
        if symbol_sequence[i] != transitted_sequence_combined[i]:
            SERerrors += 1

    SER_begin = SERerrors / len(symbol_sequence)
    SER_log = None if SER_begin == 0 else math.log10(SER_begin)
    SER = SER_log
    bit_sequence_combined = assist_symbol_to_bit(transitted_sequence_combined, MAP_4QAM)

    for i in range(len(bit_sequence)):
        if bit_sequence[i] != bit_sequence_combined[i]:
            BERerrors += 1

    BER_begin = BERerrors / len(bit_sequence)
    BER_log = None if BER_begin == 0 else math.log10(BER_begin)
    BER = BER_log

    return noisy_symbol_blocks, SER, BER


def SER_BER_4QAM_DFE_DYNAMIC(
    bit_sequence: BIT_SEQUENCE_TYPE,
    block_size: int,
    random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE,
    random_values_for_CIR: RANDOM_VALUES_CIR_TYPE,
    snr: float,
) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using 4QAM
        - splits the symbol sequence into blocks with the given block size
        - calculate a channel impulse response for each block using the dynamic impulse response equation in the practical guide
        - add noise to the symbol sequence in each block using the equation in the practical guide and the dynamic impulse response and SNR
        - save the noisy symbol sequence from each block
        - perform the DFE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        random_values_for_CIR -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_CIR[0][0][0] and returns a float.
            These random values are used to calculate the dynamic impulse response using equation 4 in the practical guide
            The first index indicates the corresponding block. The second index indicate the corresponding coeficient (c0 - [0], c1 = [1], c2 - [2]). The third index indicates the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_CIR[7][2][1] -> is the float that should be used for the imaginary component kappa value to calculate the c2 coeficient for the 8th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.

    """
    symbol_sequence = assist_bit_to_symbol(bit_sequence, MAP_4QAM)
    transmitted_sequence = []
    blocks = assist_split_symbols_into_blocks(symbol_sequence, block_size)
    noisy_symbol_blocks = []

    sigma = 1 / cm.sqrt(10 ** (snr / 10) * 2)
    SERerrors = 0
    BERerrors = 0
    cdynamic = []

    for block_index in range(len(blocks)):
        noisy_symbols = []
        blocks[block_index] = [
            (0.7071067811865475 + 0.7071067811865475j),
            (0.7071067811865475 + 0.7071067811865475j),
        ] + blocks[block_index]
        for symbol_index in range(2, len(blocks[block_index])):

            cdynamic = [
                (
                    random_values_for_CIR[block_index][0][0]
                    + 1j * random_values_for_CIR[block_index][0][1]
                )
                / cm.sqrt(6),
                (
                    random_values_for_CIR[block_index][1][0]
                    + 1j * random_values_for_CIR[block_index][1][1]
                )
                / cm.sqrt(6),
                (
                    random_values_for_CIR[block_index][2][0]
                    + 1j * random_values_for_CIR[block_index][2][1]
                )
                / cm.sqrt(6),
            ]
            noise_values = (
                random_values_for_symbols[block_index][symbol_index - 2][0]
                + (random_values_for_symbols[block_index][symbol_index - 2][1]) * 1j
            ) / cm.sqrt(2)

            noise_val = sigma * noise_values

            rt = (
                blocks[block_index][symbol_index] * cdynamic[0]
                + blocks[block_index][symbol_index - 1] * cdynamic[1]
                + blocks[block_index][symbol_index - 2] * cdynamic[2]
                + noise_val
            )

            noisy_symbols.append(rt)
        noisy_symbol_blocks.append(noisy_symbols)

    for i in range(len(noisy_symbol_blocks)):
        cdynamic = [
            (random_values_for_CIR[i][0][0] + 1j * random_values_for_CIR[i][0][1])
            / cm.sqrt(6),
            (random_values_for_CIR[i][1][0] + 1j * random_values_for_CIR[i][1][1])
            / cm.sqrt(6),
            (random_values_for_CIR[i][2][0] + 1j * random_values_for_CIR[i][2][1])
            / cm.sqrt(6),
        ]

        transmitted = DFE_4QAM_BLOCK(noisy_symbol_blocks[i], cdynamic)

        transmitted_sequence.append(transmitted)

    transitted_sequence_combined = assist_combine_blocks_into_symbols(
        transmitted_sequence
    )

    for i in range(len(symbol_sequence)):
        if symbol_sequence[i] != transitted_sequence_combined[i]:
            SERerrors += 1

    SER_begin = SERerrors / len(symbol_sequence)
    SER_log = None if SER_begin == 0 else math.log10(SER_begin)
    SER = SER_log
    bit_sequence_combined = assist_symbol_to_bit(transitted_sequence_combined, MAP_4QAM)

    for i in range(len(bit_sequence)):
        if bit_sequence[i] != bit_sequence_combined[i]:
            BERerrors += 1

    BER_begin = BERerrors / len(bit_sequence)
    BER_log = None if BER_begin == 0 else math.log10(BER_begin)
    BER = BER_log

    return noisy_symbol_blocks, SER, BER


def SER_BER_4QAM_MLSE_STATIC(
    bit_sequence: BIT_SEQUENCE_TYPE,
    block_size: int,
    random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE,
    snr: float,
) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using 4QAM
        - splits the symbol sequence into blocks with the given block size
        - add the appended symbols to each sequence in each block. For 4QAM the appended symbols are [0.7071067811865475+0.7071067811865475j, 0.7071067811865475+0.7071067811865475j]
        - add noise to the symbol sequence including the appended symbols in each block using the equation in the practical guide and the static impulse response and SNR.
        - save the noisy symbol sequence from each block which should all be block size + 2 long, thus including the symbols that was appended.
        - perform the MLSE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences excluding the appended symbols.
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.

    """
    CIR = [0.29 + 0.98j, 0.73 - 0.24j, 0.21 + 0.91j]
    symbol_sequence = assist_bit_to_symbol(bit_sequence, MAP_4QAM)
    transmitted_sequence = []
    blocks = assist_split_symbols_into_blocks(symbol_sequence, block_size)
    noisy_symbol_blocks = []

    sigma = 1 / cm.sqrt(10 ** (snr / 10) * 2)
    SERerrors = 0
    BERerrors = 0

    for block_index in range(len(blocks)):
        noisy_symbols = []
        blocks[block_index] = (
            [
                (0.7071067811865475 + 0.7071067811865475j),
                (0.7071067811865475 + 0.7071067811865475j),
            ]
            + blocks[block_index]
            + [
                (0.7071067811865475 + 0.7071067811865475j),
                (0.7071067811865475 + 0.7071067811865475j),
            ]
        )
        for symbol_index in range(2, len(blocks[block_index])):

            noise_values = (
                random_values_for_symbols[block_index][symbol_index - 2][0]
                + (random_values_for_symbols[block_index][symbol_index - 2][1]) * 1j
            ) / cm.sqrt(2)

            noise_val = sigma * noise_values

            rt = (
                blocks[block_index][symbol_index] * CIR[0]
                + blocks[block_index][symbol_index - 1] * CIR[1]
                + blocks[block_index][symbol_index - 2] * CIR[2]
                + noise_val
            )

            noisy_symbols.append(rt)
        noisy_symbol_blocks.append(noisy_symbols)

    for i in range(len(noisy_symbol_blocks)):

        transmitted = MLSE_4QAM_BLOCK(noisy_symbol_blocks[i], CIR)

        transmitted_sequence.append(transmitted)

    transitted_sequence_combined = assist_combine_blocks_into_symbols(
        transmitted_sequence
    )

    for i in range(len(symbol_sequence)):
        if symbol_sequence[i] != transitted_sequence_combined[i]:
            SERerrors += 1

    SER_begin = SERerrors / len(symbol_sequence)
    SER_log = None if SER_begin == 0 else math.log10(SER_begin)
    SER = SER_log
    bit_sequence_combined = assist_symbol_to_bit(transitted_sequence_combined, MAP_4QAM)

    for i in range(len(bit_sequence)):
        if bit_sequence[i] != bit_sequence_combined[i]:
            BERerrors += 1

    BER_begin = BERerrors / len(bit_sequence)
    BER_log = None if BER_begin == 0 else math.log10(BER_begin)
    BER = BER_log

    return noisy_symbol_blocks, SER, BER


def SER_BER_4QAM_MLSE_DYNAMIC(
    bit_sequence: BIT_SEQUENCE_TYPE,
    block_size: int,
    random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE,
    random_values_for_CIR: RANDOM_VALUES_CIR_TYPE,
    snr: float,
) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using BPSK
        - splits the symbol sequence into blocks with the given block size
        - add the appended symbols to each sequence in each block. For BPSK the appended symbols are [0.7071067811865475+0.7071067811865475j, 0.7071067811865475+0.7071067811865475j]
        - calculate a channel impulse response for each block using the dynamic impulse response equation in the practical guide
        - add noise to the symbol sequence including the appended symbols in each block using the equation in the practical guide and the static impulse response and SNR.
        - save the noisy symbol sequence from each block which should all be block size + 2 long, thus including the symbols that was appended.
        - perform the MLSE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences excluding the appended symbols.
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        random_values_for_CIR -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_CIR[0][0][0] and returns a float.
            These random values are used to calculate the dynamic impulse response using equation 4 in the practical guide
            The first index indicates the corresponding block. The second index indicate the corresponding coeficient (c0 - [0], c1 = [1], c2 - [2]). The third index indicates the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_CIR[7][2][1] -> is the float that should be used for the imaginary component kappa value to calculate the c2 coeficient for the 8th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.

    """
    symbol_sequence = assist_bit_to_symbol(bit_sequence, MAP_4QAM)
    transmitted_sequence = []
    blocks = assist_split_symbols_into_blocks(symbol_sequence, block_size)
    noisy_symbol_blocks = []

    sigma = 1 / cm.sqrt(10 ** (snr / 10) * 2)
    SERerrors = 0
    BERerrors = 0
    cdynamic = []

    for block_index in range(len(blocks)):
        noisy_symbols = []
        blocks[block_index] = (
            [
                (0.7071067811865475 + 0.7071067811865475j),
                (0.7071067811865475 + 0.7071067811865475j),
            ]
            + blocks[block_index]
            + [
                (0.7071067811865475 + 0.7071067811865475j),
                (0.7071067811865475 + 0.7071067811865475j),
            ]
        )
        for symbol_index in range(2, len(blocks[block_index])):

            cdynamic = [
                (
                    random_values_for_CIR[block_index][0][0]
                    + 1j * random_values_for_CIR[block_index][0][1]
                )
                / cm.sqrt(6),
                (
                    random_values_for_CIR[block_index][1][0]
                    + 1j * random_values_for_CIR[block_index][1][1]
                )
                / cm.sqrt(6),
                (
                    random_values_for_CIR[block_index][2][0]
                    + 1j * random_values_for_CIR[block_index][2][1]
                )
                / cm.sqrt(6),
            ]
            noise_values = (
                random_values_for_symbols[block_index][symbol_index - 2][0]
                + (random_values_for_symbols[block_index][symbol_index - 2][1]) * 1j
            ) / cm.sqrt(2)

            noise_val = sigma * noise_values

            rt = (
                blocks[block_index][symbol_index] * cdynamic[0]
                + blocks[block_index][symbol_index - 1] * cdynamic[1]
                + blocks[block_index][symbol_index - 2] * cdynamic[2]
                + noise_val
            )

            noisy_symbols.append(rt)
        noisy_symbol_blocks.append(noisy_symbols)

    for i in range(len(noisy_symbol_blocks)):
        cdynamic = [
            (random_values_for_CIR[i][0][0] + 1j * random_values_for_CIR[i][0][1])
            / cm.sqrt(6),
            (random_values_for_CIR[i][1][0] + 1j * random_values_for_CIR[i][1][1])
            / cm.sqrt(6),
            (random_values_for_CIR[i][2][0] + 1j * random_values_for_CIR[i][2][1])
            / cm.sqrt(6),
        ]

        transmitted = MLSE_4QAM_BLOCK(noisy_symbol_blocks[i], cdynamic)

        transmitted_sequence.append(transmitted)

    transitted_sequence_combined = assist_combine_blocks_into_symbols(
        transmitted_sequence
    )

    for i in range(len(symbol_sequence)):
        if symbol_sequence[i] != transitted_sequence_combined[i]:
            SERerrors += 1

    SER_begin = SERerrors / len(symbol_sequence)
    SER_log = None if SER_begin == 0 else math.log10(SER_begin)
    SER = SER_log
    bit_sequence_combined = assist_symbol_to_bit(transitted_sequence_combined, MAP_4QAM)

    for i in range(len(bit_sequence)):
        if bit_sequence[i] != bit_sequence_combined[i]:
            BERerrors += 1

    BER_begin = BERerrors / len(bit_sequence)
    BER_log = None if BER_begin == 0 else math.log10(BER_begin)
    BER = BER_log

    return noisy_symbol_blocks, SER, BER


####### DO NOT EDIT #######
if __name__ == "__main__":

    evaluate()
####### DO NOT EDIT #######
