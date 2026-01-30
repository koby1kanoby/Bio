# Simulation of storing a bit sequence in DNA using a 6-letter alphabet
# Assumptions (since Table 2 is not explicitly given):
# Alphabet: {A, T, G, C, R, Y}
# R -> {A,G} with equal probability, Y -> {C,T} with equal probability
# Each word is length 3 over this alphabet -> encodes 6 bits (2 bits per letter)
# Mapping table is explicitly defined below

import random
from collections import defaultdict, Counter
import difflib

WORD_LEN = 2          # letters per word
BITS_PER_WORD = 5

# choose d1, d2
# d1, d2 = 2, 3
d1, d2 = 1, 2

bit_data = '000111111000101100001000000010000010111000010010100111100110000100000000000000110011000000110000001100010010111001101010011100000110011100000000001011000111101001111011001110001101001111111111111111111111111011100100100110000100001001101010110010100110000111001000001000010011011010001010001100000110010111001110111000110110110110011000111000111110011010011000110100111011101100010000010001100110010011100110001011000111001000101000011010101100110100010000010100101100101100101001110110110001110001100010100100111001000001000011100101001101001000011100111010100011110110000111001110010001010100010001101000110000100100100001001110011110011011010100011011111001111111101111111101010111101110111101111111100111110000111101101011111110011111110101011110000011111010011110111101110111110100111111111011101111101110111010010111100111111100111111111111111001110100000110100111000100000000100010101011000001000100000000000111100001111000000000100000001000011111111011000000101111111001001110000000001100110000000001110001111000010010000100100001001000010100000100100011110000100100011111000101010001011000111001011111100100110001000100000111000111010001010010010111000100110001001100010111000101000101001010000110100000100010000010001010001010101110101011100000011001010101010101010101010010111101101000100110110101111010111110011100001101000111101000101000101010101010101010100111100010110111001100110010001101110011000010110010101101101001001010111111011001001010000011100010110011110111000010110110011101010011010010110010101110001001101111000001110111011100001011100100010111010011111110110011110011110000111100111000101000000010111111000111110000000011110010110111001100110111111001011110011111111011101111000000011111110011111111100001111110011100001000001011110001111110111111100000001010000011010000010100011110000101100011110111000000010110001101001001110000000001000111011110000111100001111001100011111111000011110001'

SEGMENT_LENGTH = BITS_PER_WORD * 2

SEGMENT_COUNT = int(len(bit_data) / SEGMENT_LENGTH)

DROPLETS_PER_SEED = 100

# Convert bits to words
bit_to_word = {
    '00000': 'AA', '00001': 'AC', '00010': 'AT', '00011': 'AG', '00100': 'AX', '00101': 'AY',
    '00110': 'CA', '00111': 'CC', '01000': 'CT', '01001': 'CG', '01010': 'CX', '01011': 'CY',
    '01100': 'TA', '01101': 'TC', '01110': 'TT', '01111': 'TG', '10000': 'TX', '10001': 'TY',
    '10010': 'GA', '10011': 'GC', '10100': 'GT', '10101': 'GG', '10110': 'GX', '10111': 'GY',
    '11000': 'XA', '11001': 'XC', '11010': 'XT', '11011': 'XG', '11100': 'XX', '11101': 'XY',
    '11110': 'YA', '11111': 'YC'
}

# Convert words to bits
word_to_bit = {v: k for k, v in bit_to_word.items()}

# Convert letters to DNA
letter_expansion = {
    'A': ['A'], 'C': ['C'], 'G': ['G'], 'T': ['T'],
    'X': ['A', 'G'],
    'Y': ['C', 'T']
}


# Function to convert bit string to word string
def bits_to_words(bits: str) -> str:
    words = ''
    for i in range(0, len(bits), BITS_PER_WORD):
        chunk = bits[i:i+BITS_PER_WORD]
        words = words + bit_to_word[chunk]
    return words


# Function to convert word string to bit string
def words_to_bits(words: str) -> str:
    bits = ''
    for i in range(0, len(words), WORD_LEN):
        chunk = words[i:i + WORD_LEN]
        bits = bits + word_to_bit[chunk]
    return bits


# Get the correct DNA element based on distribution and error cahnce
def expand_word_to_dna(word, error_chance=0.99):
    dna = ''
    for letter in word:
        used_letter = letter if random.random() < error_chance else random.choice(list(letter_expansion.keys()))
        dna = dna + random.choice(letter_expansion[used_letter])
    return dna


# TODO: Provide unique barcodes somehow
def random_barcode(exclusion_list, is_d1=True, length=10):
    while True:
        barcode = ''.join(random.choice('ATGC') for _ in range(length-1)) + ('A' if is_d1 else 'T')
        if barcode not in exclusion_list:
            return barcode


# Get the set segments from a given seed
def get_values_from_seed(seed_value: str, length: int, max_value: int):
    random.seed(seed_value)

    sequence = [random.randint(0, max_value-1) for _ in range(length)]
    return sequence


# Get XOR product of the segments
def mod_two_from_segments(segments):
    product = ''
    if len(segments) == 0:
        return None
    for index in range(0, len(segments[0])):
        product = product + str(sum([int(segment[index]) for segment in segments]) % 2)
    return product


# Function to cut out the correct segments
def get_segments_by_index(data: str, indexes):
    segments = [
        data[i:i + SEGMENT_LENGTH]
        for i in range(0, len(data), SEGMENT_LENGTH)
    ]

    return [segments[i] for i in indexes if i < len(segments)]


# Decipher DNA data to words
def decipher_dna(dna_seqs, length_of_seq):
    decoded_payload = ''

    # FOr each index
    for index in range(0, length_of_seq):
        current_letter = ''
        index_dict = {
            'A': 0,
            'C': 0,
            'T': 0,
            'G': 0
        }
        # For each sequence in the data
        for seq in dna_seqs:
            letter_at_index = seq[index]
            index_dict[letter_at_index] = index_dict[letter_at_index] + 1

        # If we are certain this is it then assign
        for key in index_dict.keys():
            if (index_dict[key] > 90):
                current_letter = key

        # If there is even distribution of these assign X or Y
        if index_dict['A'] > 30 and index_dict['G'] > 30:
            current_letter = 'X'
        elif index_dict['C'] > 30 and index_dict['T'] > 30:
            current_letter = 'Y'

        decoded_payload = decoded_payload + current_letter
    return decoded_payload


# Function to encode the bit data as genetic information
def encode(bits):
    droplets = []
    barcodes = {}

    # For each seed generate droplets
    for seed in range(0, 1024):
        seed_bits = bin(seed)[2:].zfill(SEGMENT_LENGTH)
        length = d1 if random.random() < 0.5 else d2  # Random choice of d1 and d2
        curr_segment_numbers = get_values_from_seed(seed_bits, length, int(SEGMENT_COUNT))  # get the segment indexes from the seed
        barcode = random_barcode(barcodes, length == d1)  # Generate a new unique random barcode
        barcodes[barcode] = seed  # Save it
        segments = get_segments_by_index(bits, curr_segment_numbers)  # Get the actual segment data from the indexes
        xor_of_segments = mod_two_from_segments(segments)  # Get mod 2 of bit data

        # Fail safe
        if len(segments) == 0:
            print(curr_segment_numbers)
            print(seed_bits)
            print(drop_num)

        # Get words in letter form from XOR of bits
        words_of_xor_bits = bits_to_words(xor_of_segments)


        # Generate many sequences at correct distribution
        for drop_num in range(0, DROPLETS_PER_SEED):
            translated_sub_segments = expand_word_to_dna(words_of_xor_bits)

            # Get DNA data from seed value
            dna_of_seed = expand_word_to_dna(bits_to_words(seed_bits))

            # TODO: Reject homogeneous droplet data

            # Save the final full sequence of droplet data
            droplet = barcode + dna_of_seed + translated_sub_segments
            droplets.append(droplet)

    # Randomly shuffle all sequences (like mixing test tube)
    random.shuffle(droplets)
    return droplets, barcodes


# Function to decode the original bit data from genetic sequences
def decode(sequences):
    groups = defaultdict(list)
    decoded_droplets = {}
    seeds_of_barcodes = defaultdict(list)
    decoded_seeds = {}

    # For each sequence make dict of payloads and seeds by barcode
    for seq in sequences:
        barcode = seq[:10]
        payload = seq[14:]
        seed = seq[10:14]
        groups[barcode].append(payload)
        seeds_of_barcodes[barcode].append(seed)

    # Decipher the payload and seed data for each barcode
    for barcode, payloads in groups.items():
        decoded_droplets[barcode] = decipher_dna(payloads, 4)

    for barcode, seeds in seeds_of_barcodes.items():
        decoded_seeds[barcode] = decipher_dna(seeds, 4)

    barcodes = list(decoded_droplets.keys())

    # Sort by the last character so that the ones which represent a single segment come first
    barcodes.sort(key=lambda x: x[-1])

    deciphered_segments = [''] * SEGMENT_COUNT

    segments = []

    # Decipher original data from decoded paylaods
    for barcode in barcodes:
        decoded_droplet = decoded_droplets[barcode]
        seed = decoded_seeds[barcode]
        seed_bits = words_to_bits(seed)
        # For those which represent a single segment
        if barcode[-1] == 'A':
            segment_numbers = get_values_from_seed(seed_bits, d1, int(SEGMENT_COUNT))

            for seg in segment_numbers:
                if seg not in segments:
                    segments.append(seg)

            deciphered_segments[segment_numbers[0]] = words_to_bits(decoded_droplet)
        # Barcode is for multiple segments
        else:
            segment_numbers = get_values_from_seed(seed_bits, d2, int(SEGMENT_COUNT))

            for seg in segment_numbers:
                if seg not in segments:
                    segments.append(seg)

            first_segment = segment_numbers[0]
            second_segment = segment_numbers[1]

            if deciphered_segments[first_segment] != '' and deciphered_segments[second_segment] != '':
                continue
            elif deciphered_segments[first_segment] != '' and deciphered_segments[second_segment] == '':
                known_segment = deciphered_segments[first_segment]
                missing_index = second_segment
            elif deciphered_segments[first_segment] == '' and deciphered_segments[second_segment] != '':
                known_segment = deciphered_segments[second_segment]
                missing_index = first_segment
            else:
                continue
                #TODO: Make another go at these

            if missing_index == 34 or missing_index == 174:
                print(missing_index)
                print(known_segment)
                print(words_to_bits(decoded_droplet))
                print(mod_two_from_segments([known_segment, words_to_bits(decoded_droplet)]))

            deciphered_segments[missing_index] = mod_two_from_segments([known_segment, words_to_bits(decoded_droplet)])

    deciphered_segments = [item for item in deciphered_segments if item != '']

    seed_bits_list = []

    for seed in list(decoded_seeds):
        seed_bits_list.append(words_to_bits(seed))

    seed_bits_list.sort()

    return ''.join(deciphered_segments)


# Main function to simulate encoding and decoding of data
def main():
    droplets, barcodes = encode(bit_data)

    print('encoding done')

    recovered = decode(droplets)

    print('decoding done')

    if recovered == bit_data:
        print('Recovered the original data')
    else:
        correct = 0
        wrong = 0
        for i in range(SEGMENT_COUNT):
            rec = get_segments_by_index(recovered, [i])[0]
            orig = get_segments_by_index(bit_data, [i])[0]
            if (rec != orig):
                wrong += 1
            else:
                correct += 1
        print('correct percentage')
        print(correct / (correct + wrong))


if __name__ == '__main__':
    main()
