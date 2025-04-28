import cmath as cm
import numpy as np


#########################################################################
#                       Assisting functions                             #
#########################################################################


def convert_message_to_bits(message: str) -> list:
    """
    Converts a message string to a sequence of bits using ASCII table. Each letter produces 8 bits / 1 byte

    parameters:
        message -> type <class 'str'> : A string containing text

    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """
    bits = []

    for char in message:
        ascii_value = ord(char)  # gives value in decimal
        binary_value = format(ascii_value, "08b")  # changing it to be in binary

        for bit in binary_value:
            bit = int(bit)
            bits.append(bit)

    return bits


#########################################################################
#                       Question 1: Lempel-Ziv                          #
#########################################################################


def lempel_ziv_calculate_dictionary_table(bit_sequence: list) -> list:
    """
    Uses a sequence of bits to determine the dictionary table which can be used to compress a sequence of bits

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]

    returns:
        lempel_ziv_dictionary_table -> type <class 'list'> : A list containing lists. Each list entry contains three bit sequences, the dictionary location, the dictionary phrase, and the codeword, in that order.
            For example, the first few rows in the lecture notes:

            [
                [[0, 0, 0, 0, 1], [1], [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 0], [0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 1, 1], [1, 0], [0, 0, 0, 0, 1, 0]],
                [[0, 0, 1, 0, 0], [1, 1], [0, 0, 0, 0, 1, 1]],
            ]

            Do not sort the array
    """

    # Getting the dictionary phrase
    dictionary_phrase = []
    dictionary_location = []
    dummy_location = []
    dictionary_table = []

    seen = []
    currentstring = []

    for i in bit_sequence:
        currentstring.append(i)
        if currentstring not in seen:
            dictionary_phrase.append(currentstring)
            seen.append(currentstring)
            currentstring = []

    counter = len(dictionary_phrase)

    # Generate dictionary locations

    count = 1
    dummy_location = []
    while count <= counter:
        count_binary = format(count, "b")
        dummy_location.append(count_binary)
        count += 1

    binary_length = format(len(dictionary_phrase), "b")
    countletters = len(binary_length)

    for i in dummy_location:
        if len(i) < countletters:
            sub = countletters - len(i)
            j = "0" * sub + i  # Pad with leading zeros
        else:
            j = i

        # Convert the padded binary string to a list of integers
        binary_list = []
        for bit in j:
            binary_list.append(int(bit))

        # Add the list of integers to dictionary_location
        dictionary_location.append(binary_list)

    codeword = []

    # Generate codewords
    for idx in range(len(dictionary_phrase)):
        if len(dictionary_phrase[idx]) == 1:
            codeword.append([0] * countletters + dictionary_phrase[idx])
        else:
            modified_phrase = dictionary_phrase[idx][:-1]
            last_char = dictionary_phrase[idx][-1]
            for j in range(len(dictionary_phrase)):
                if modified_phrase == dictionary_phrase[j]:
                    location_code = dictionary_location[j]
                    codeword_bits = list(map(int, location_code)) + [int(last_char)]
                    codeword.append(codeword_bits)
                    break

    # Combine dictionary_location, dictionary_phrase, and codeword into one list
    for i in range(len(dictionary_phrase)):

        entry = [
            dictionary_location[i],
            dictionary_phrase[i],
            codeword[i],
        ]
        dictionary_table.append(entry)

    return dictionary_table


def lempel_ziv_compress_bit_sequence(
    bit_sequence: list, lempel_ziv_dictionary_table: list
) -> list:
    """
    Compresses a sequence of bits using the lempel-ziv algorithm and the lempel ziv codewords in the lempel ziv dictionary table

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        lempel_ziv_dictionary_table -> type <class 'list'> : A list containing lists. Each list entry contains three bit sequences, the dictionary location, the dictionary phrase, and the codeword, in that order.
            See example at function lempel_ziv_calculate_dictionary_table

    returns:

     compressed_bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """

    seq = [bit_sequence[0]]
    compressed = []

    for i in range(1, len(bit_sequence)):

        match_found = False
        seq.append(bit_sequence[i])

        # Check if the current seq is in the dictionary
        for entry in lempel_ziv_dictionary_table:
            if seq == entry[1]:
                match_found = True
                break

        if not match_found:
            # If no match was found, remove the last bit and check again
            last_bit = seq.pop()
            for entry in lempel_ziv_dictionary_table:
                if seq == entry[1]:
                    compressed.extend(entry[2])
                    seq = [last_bit]  # Reset seq with the last bit
                    break

    while seq:
        match_found = False
        for entry in lempel_ziv_dictionary_table:
            if seq == entry[1]:
                compressed.extend(entry[2])
                seq = []
                match_found = True
                break
        if not match_found:
            seq.pop()  # Remove the last bit if no match is found

    return compressed


def lempel_ziv_decompress_bit_sequence(
    compressed_bit_sequence: list, lempel_ziv_dictionary_table: list
) -> list:
    """
    Decompress a sequence of bits using the lempel-ziv algorithm and the lempel-ziv codewords in the lempel-ziv dictionary table

    parameters:
        compressed_bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        lempel_ziv_dictionary_table -> type <class 'list'> : A list containing lists. Each list entry contains three bit sequences, the dictionary location, the dictionary phrase, and the codeword, in that order.
            See example at function lempel_ziv_calculate_dictionary_table

    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]

    """
    const_length = len(lempel_ziv_dictionary_table[0][2])

    decompressed = []

    for i in range(0, len(compressed_bit_sequence), const_length):
        codeword = compressed_bit_sequence[i : i + const_length]

        for j in lempel_ziv_dictionary_table:
            if codeword == j[2]:
                decompressed.extend(j[1])

                break

    return decompressed


#########################################################################
#                       Question 2: Huffman coding                      #
#########################################################################


def huffman_get_mapping_codes() -> list:
    """
    Returns the mapping codes generated using a huffman coding tree, which can be used to compress a message

    parameters:
        None -> The tree can be calculated by hand and this function just returns the result

    returns:
        huffman_mapping -> type <class 'list'> : A list containing lists. Each list entry contains a string and a bit sequence, the letter as a string, and the corresponding bit sequence.
            For example, the huffman mapping codes for example 1 in the lecture notes:

            [
                ['x_1', [0, 0]],
                ['x_2', [0, 1]],
                ['x_3', [1, 0]],
                ['x_4', [1, 1, 0]],
                ['x_5', [1, 1, 1, 0]],
                ['x_6', [1, 1, 1, 1, 0]],
                ['x_7', [1, 1, 1, 1, 1]]
            ]
    """

    huffman_list = [
        ["a", [0, 0, 0, 0]],
        ["b", [1, 1, 0, 0]],
        ["c", [0, 0, 1, 1]],
        ["d", [0, 1, 1, 1]],
        ["e", [0, 0, 0, 1, 0]],
        ["f", [0, 0, 1, 0, 0, 0]],
        ["g", [1, 1, 1, 1, 1, 0, 1, 1, 0]],
        ["h", [1, 1, 1, 0, 0]],
        ["i", [0, 1, 1, 0, 0]],
        ["j", [0, 0, 0, 1, 1]],
        ["k", [0, 1, 0, 0, 0]],
        ["l", [1, 1, 1, 1, 1, 0, 0]],
        ["m", [0, 1, 0, 0, 1, 1]],
        [" ", [1, 0]],
        ["n", [0, 0, 1, 0, 1]],
        ["o", [1, 1, 0, 1, 1, 0]],
        ["p", [1, 1, 1, 1, 1, 1]],
        ["q", [0, 1, 0, 1, 0]],
        ["r", [0, 1, 0, 1, 1]],
        ["s", [1, 1, 0, 1, 0]],
        ["t", [1, 1, 1, 0, 1]],
        ["u", [0, 1, 1, 0, 1]],
        ["v", [1, 1, 0, 1, 1, 1]],
        ["w", [1, 1, 1, 1, 1, 0, 1, 0]],
        ["x", [0, 1, 0, 0, 1, 0]],
        ["y", [1, 1, 1, 1, 0]],
        ["z", [0, 0, 1, 0, 0, 1]],
        [".", [1, 1, 1, 1, 1, 0, 1, 1, 1]],
    ]

    return huffman_list


def huffman_compress_message(message: str, huffman_mapping: list) -> list:
    """
    Compresses a text message using the huffman mapping codes generated in function huffman_get_mapping_codes and generates a sequence of bits. Assume input consists of only characters within the huffman mapping codes.

    parameters:
        message -> type <class 'str'> : A string containing text
        huffman_mapping -> type <class 'list'> : A list containing lists. Each list entry contains a string and a bit sequence, the letter as a string, and the corresponding bit sequence.
            See example at function huffman_get_mapping_codes

    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
    """
    lower_caps = message.lower()

    bit_sequence = []
    bit_seq = []

    for char in lower_caps:
        for i in huffman_mapping:
            if char == i[0]:
                bit_sequence.append(i[1])
                break

    for i in bit_sequence:
        for j in i:
            bit_seq.append(j)

    return bit_seq


def huffman_decompress_bit_sequence(bit_sequence: list, huffman_mapping: list) -> list:
    """
    Decompresses a text message using the huffman mapping codes generated in function huffman_get_mapping_codes and generates single text message.

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        huffman_mapping -> type <class 'list'> : A list containing lists. Each list entry contains a string and a bit sequence, the letter as a string, and the corresponding bit sequence.
            See example at function huffman_get_mapping_codes

    returns:
        message -> type <class 'str'> : A string containing text
    """

    message = ""
    current_bits = []
    for i in bit_sequence:
        current_bits.append(i)
        for j in huffman_mapping:
            if current_bits == j[1]:
                message += j[0]
                current_bits = []
                break

    return message


#########################################################################
#                       Question 3: Simulation Platform                 #
#########################################################################


def modulation_get_symbol_mapping(scheme: str) -> list:
    """
    Returns the bit to symbol mapping for the given scheme. Returns NoneType if provided scheme name is not available.
    Required scheme implementations: "BPSK", "4QAM", "8PSK", "16QAM".

    parameters:
        scheme -> type <class 'str'> : A string containing the name of the scheme for which a bit to symbol mapping needs to be returned.

    returns:
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
        For example:
            [
                [ 1+1j,    [0, 0] ],
                [ -1+1j,   [0, 1] ],
                [ -1-1j,   [1, 1] ],
                [ 1-1j,    [1, 0] ]
            ]
    """

    BPSK = []
    QAM = []
    PSK = []
    QAM16 = []

    if scheme == "BPSK":
        BPSK = [[(-1), [0]], [(1), [1]]]
        return BPSK

    elif scheme == "4QAM":
        QAM = [
            [(1 + 1j) / cm.sqrt(2), [0, 0]],
            [(-1 + 1j) / cm.sqrt(2), [0, 1]],
            [(-1 - 1j) / cm.sqrt(2), [1, 1]],
            [(1 - 1j) / cm.sqrt(2), [1, 0]],
        ]
        return QAM

    elif scheme == "8PSK":
        PSK = [
            [(-1 - 1j) / cm.sqrt(2), [0, 0, 0]],
            [-1, [0, 0, 1]],
            [1j, [0, 1, 0]],
            [(-1 + 1j) / cm.sqrt(2), [0, 1, 1]],
            [-1j, [1, 0, 0]],
            [(1 - 1j) / cm.sqrt(2), [1, 0, 1]],
            [(1 + 1j) / cm.sqrt(2), [1, 1, 0]],
            [1, [1, 1, 1]],
        ]
        return PSK

    elif scheme == "16QAM":
        QAM16 = [
            [(-3 + 3j) / (3 * cm.sqrt(2)), [0, 0, 0, 0]],
            [(-3 + 1j) / (3 * cm.sqrt(2)), [0, 0, 0, 1]],
            [(-3 - 3j) / (3 * cm.sqrt(2)), [0, 0, 1, 0]],
            [(-3 - 1j) / (3 * cm.sqrt(2)), [0, 0, 1, 1]],
            [(-1 + 3j) / (3 * cm.sqrt(2)), [0, 1, 0, 0]],
            [(-1 + 1j) / (3 * cm.sqrt(2)), [0, 1, 0, 1]],
            [(-1 - 3j) / (3 * cm.sqrt(2)), [0, 1, 1, 0]],
            [(-1 - 1j) / (3 * cm.sqrt(2)), [0, 1, 1, 1]],
            [(3 + 3j) / (3 * cm.sqrt(2)), [1, 0, 0, 0]],
            [(3 + 1j) / (3 * cm.sqrt(2)), [1, 0, 0, 1]],
            [(3 - 3j) / (3 * cm.sqrt(2)), [1, 0, 1, 0]],
            [(3 - 1j) / (3 * cm.sqrt(2)), [1, 0, 1, 1]],
            [(1 + 3j) / (3 * cm.sqrt(2)), [1, 1, 0, 0]],
            [(1 + 1j) / (3 * cm.sqrt(2)), [1, 1, 0, 1]],
            [(1 - 3j) / (3 * cm.sqrt(2)), [1, 1, 1, 0]],
            [(1 - 1j) / (3 * cm.sqrt(2)), [1, 1, 1, 1]],
        ]
        return QAM16

    return None


def modulation_map_bits_to_symbols(bit_to_symbol_map: list, bit_sequence: list) -> list:
    """
    Returns a sequence of symbols that corresponds to the provided sequence of bits using the bit to symbol map that respresent the modulation scheme

    parameters:
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
            See example at function modulation_get_symbol_mapping
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]

    returns:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j]

    """

    length_of_sequence = len(bit_to_symbol_map[0][1])
    symbols = []

    for i in range(0, len(bit_sequence), length_of_sequence):
        bit_chunk = bit_sequence[i : i + length_of_sequence]
        for j in bit_to_symbol_map:

            if bit_chunk == j[1]:
                symbols.append(j[0])
                break

    return symbols


def modulation_map_symbols_to_bits(
    bit_to_symbol_map: list, symbol_sequence: list
) -> list:
    """
    Returns a sequence of bits that corresponds to the provided sequence of symbols containing noise using the bit to symbol map that respresent the modulation scheme and the euclidean distance

    parameters:
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
            See example at function modulation_get_symbol_mapping
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j]

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


def modulation_determine_SER_and_BER(
    bit_sequence: list,
    symbol_sequence: list,
    bit_to_symbol_map: list,
    gaussian_random_values: list,
    SNR_range: list,
) -> tuple:
    """
    Returns a range of SER and BER values over the given SNR range using the supplied sequence of bits and symbols and noise calculated using the gaussian random values.

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1]
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j]
            This symbol list, is the result of the function modulation_map_bits_to_symbols using the bit_sequence and bit_to_symbol_map given for this function
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
            See example at function modulation_get_symbol_mapping
        gaussian_random_values -> type <class 'list'> : A list containing lists. Each list entry contains two floats, both random gaussian distributed values.
            These random values are used to calculate the added noise according to the equation given in the practical guide. The first number should be used for the real component, and the second number should be used for the imaginary component.
        SNR_range -> type <class 'list'> : A list containing all the SNR values for which a SER and BER should be calculated for

    returns:
        noisy_symbol_sequence -> type <class 'list'> : A list containing lists. Each list entry contains complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j]
            Since multiple SNR values are provided, a list of multiple symbol sequences should be returned for each corresponding SNR value in the SNR_range variable.
        noisy_bit_sequence - type <class 'list'> : A list containing lists. Each list entry contains int items which represents the bits for example: [0, 1, 1]
            Since multiple SNR values are provided, a list of multiple bit sequences should be returned for each corresponding SNR value in the SNR_range variable.
        SER_results -> <class 'list'> : A list containing the SER as float for the corresponding SNR value in the same index position in the SNR_range list, for example [-2.5, -5.31]
            The result should be scaled using log(SER) (base 10)
        BER_results -> <class 'list'> : A list containing the BER as float for the corresponding SNR value in the same index position in the SNR_range list, for example [-2.5, -5.31]
            The result should be scaled using log(BER) (base 10)

    Additional Explanation:

    As previously, the bit_sequence variable is a list of 1 and 0 integers.
    These bits were then converted to a list of symbols using the function
    modulation_map_bits_to_symbols, and the results were stored in the
    symbol_sequence variable. As normal the mapping between symbols and bits
    are given in the variable bit_to_symbol_map, obtained from function
    modulation_get_symbol_mapping.

    Only one bit sequence and symbol sequence are provided to the function. The
    same sequences should be used to add noise for the different SNR values.

    The gaussian_random_values variable are a list of zero mean, unity variance
    Gaussian random numbers. Each entry corresponds to a symbol in the
    symbol_sequence variable, meaning that the noise that should be added to
    symbol_sequence[4] should be gaussian_random_values[4]. Each entry in the
    list consists of two random numbers. The first random number should be used
    as the real number "n_k^(i)" from equation 1 in the practical guide. The
    second number in the entry should be used as the imaginary number "n_k^(q)"
    from equation 1 in the practical guide.

    For example, suppose the symbol list is as [(1 + 1j), ...] and the
    gaussian_random_values is as [ [1.23, -0.5] , ...]. For the first symbol
    the two noise values for the real and imaginary component are 1.23 and -0.5
    respectively, thus the symbol with noise will then be:
                (1 + 1j) + sigma * (1.23 - 0.5j) / sqrt(2)

    Remember that the equation changes for BPSK, and only the first value is used

    The list of symbols that were produced by adding the noise to the
    symbol_sequence is the noisy_symbols variable that should be returned. Since
    multiple SNR ranges are computed, multiple lists of noisy symbols should be
    returned within one list in the same position as the SNR value used for example:

        SNR_range =             [           1,                     2,                   3          ]
        noisy_symbol_sequence = [ [(1.23 - 0.5j), ...], [(0.21 - 1.2j), ...], [(0.53 + 0.2j), ...] ]

    The same is done for the noisy_bit_sequence

        noisy_bit_sequence =    [ [0, 1, 0, 0, 1, ...], [0, 1, 1, 0, 1, ...], [0, 1, 0, 1, 1, ...] ]

    Finally, the SER and BER for the different SNR values should be computed and a
    list for each is returned. Remember to scale it using log. If the value for SER and BER is zero,
    which means log10 can not be computed, then return None, type NoneType.

        SER_results =           [         -24,                   -56,                 -120         ]
        BER_results =           [         -32,                   -73,                 -185         ]

    """

    noisy_symbol_sequence = []
    noisy_bit_sequence = []
    SER_results = []
    BER_results = []

    count = len(bit_to_symbol_map)

    for SNR in SNR_range:

        sigma_squared = 1 / (2 * cm.log(count, 2) * (10 ** (SNR / 10)))
        sigma = sigma_squared**0.5

        noisy_symbols = []

        for i in range(len(symbol_sequence)):
            if count == 2:
                noisy_symbol = symbol_sequence[i] + sigma * gaussian_random_values[i][0]
                noisy_symbols.append(noisy_symbol)

            else:
                noise = complex(
                    gaussian_random_values[i][0], gaussian_random_values[i][1]
                ) / cm.sqrt(2)
                noisy_symbol = symbol_sequence[i] + sigma * noise
                noisy_symbols.append(noisy_symbol)

        noisy_symbol_sequence.append(noisy_symbols)

        detected_symbols = []
        for received_symbol in noisy_symbols:
            min_distance = 10000
            closest_symbol = None
            for mapped_symbol, bits in bit_to_symbol_map:
                distance = abs(received_symbol - mapped_symbol) ** 2
                if distance < min_distance:
                    min_distance = distance
                    closest_symbol = mapped_symbol
            detected_symbols.append(closest_symbol)

        detected_bits = []
        for detected_symbol in detected_symbols:
            for mapped_symbol, mapped_bits in bit_to_symbol_map:
                if detected_symbol == mapped_symbol:
                    detected_bits.extend(mapped_bits)
                    break
        noisy_bit_sequence.append(detected_bits)

        symbol_errors = 0

        for i in range(len(symbol_sequence)):
            if symbol_sequence[i] != detected_symbols[i]:
                symbol_errors += 1

        SER = symbol_errors / len(symbol_sequence)
        SER_log = None if SER == 0 else cm.log10(SER)

        bit_errors = 0

        for i in range(len(bit_sequence)):
            if bit_sequence[i] != detected_bits[i]:
                bit_errors += 1

        BER = bit_errors / len(bit_sequence)
        BER_log = None if BER == 0 else cm.log10(BER)

        SER_results.append(SER_log)
        BER_results.append(BER_log)

    return noisy_symbol_sequence, noisy_bit_sequence, SER_results, BER_results
