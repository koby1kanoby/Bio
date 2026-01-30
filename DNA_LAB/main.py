import random
from typing import List

bases = ['A', 'C', 'G', 'T']

complement_dict = {
    'A': 'T',
    'T': 'A',
    'C': 'G',
    'G': 'C'
}


# Get the complement of a DNA string
def get_complement(seq: str) -> str:
    return ''.join(complement_dict[base] for base in seq)


# Class that represents a single helix strand of DNA
class DNAstrand:
    def __init__(self, sequence: str, is_double: bool = False):
        self.sequence = sequence.upper()
        if is_double:
            self.upper_sequence = get_complement(self.sequence)
        self.is_double = is_double

    @property
    def length(self) -> int:
        return len(self.sequence)

    def __repr__(self):
        return f"DNA({self.sequence})"

    # Modify upper sequence with guards
    def modify_second_strand(self, new_strand):
        if new_strand == '' or None:
            self.is_double = False
            self.upper_sequence = ''
        else:
            self.is_double = True
            self.upper_sequence = new_strand

    # Checks if the strand can form a double helix with another given strand
    def can_form_double_helix(self, second_strand: "DNAstrand") -> bool:
        if self.is_double != second_strand.is_double:
            # One is double and the other is not
            return False
        elif self.is_double:
            return self.double_strand_can_form_double_helix(second_strand)
        print('single')
        return self.single_strand_can_form_double_helix(second_strand.sequence)

    # Checks if the single strand can form a double helix with another given single strand
    def single_strand_can_form_double_helix(self, second_sequence: str) -> bool:
        s1, s2 = self.sequence, second_sequence

        for shift in range(-len(s2), len(s1)):
            has_match = False
            for i in range(len(s1)):
                j = i - shift
                if 0 <= j < len(s2):
                    if complement_dict[s1[i]] != s2[j]:
                        break
                    has_match = True
            if has_match:
                return True
        return False

    # Checks if a cut double strand can form a helix with another one
    def double_strand_can_form_double_helix(self, second_strand: "DNAstrand") -> bool:
        if not self.is_double:
            return False

        first_imitation = DNAstrand(self.sequence + second_strand.upper_sequence)
        second_imitation = DNAstrand(second_strand.sequence + self.upper_sequence)
        return first_imitation.single_strand_can_form_double_helix(second_imitation.sequence)

    def get_second_helix(self):
        return self.upper_sequence

    def detach_second_helix(self):
        self.upper_sequence = ''


# Class to simulate procedures of a virtual DNA lab
class VirtualLab:
    # Amplify, emulating PCR, including small probability of failure
    @staticmethod
    def amplify(
        dna_pool: List[DNAstrand],
        primer_left: str,
        primer_right: str,
        cycles: int = 10,
        p_fail: float = 0.01
    ) -> List[DNAstrand]:
        pool = dna_pool[:]

        for _ in range(cycles):
            new_copies = []
            for dna in pool:
                if primer_left in dna.sequence and primer_right in dna.sequence:
                    if random.random() > p_fail:
                        start = dna.sequence.index(primer_left)
                        end = dna.sequence.index(primer_right, start)
                        amplified = dna.sequence[start:end + len(primer_right)]
                        new_copies.append(DNAstrand(amplified))
            pool.extend(new_copies)

        return pool

    # Sequence DNA strand with small change of failure
    @staticmethod
    def sequence(dna: DNAstrand, error_rate: float = 0.00001) -> str:
        result = ""

        for base in dna.sequence:
            if random.random() < error_rate:
                result += random.choice(bases)
            else:
                result += base

        return result

    # Gel sorting process, sort strands by length
    @staticmethod
    def length_sort(dna_pool: List[DNAstrand]) -> List[DNAstrand]:
        return sorted(dna_pool, key=lambda d: d.length)

    # Use magnetic bead connected to sequence complement to extract specific samples from test tube
    @staticmethod
    def extract(
        dna_pool: List[DNAstrand],
        target_sequence: str,
        success_chance: float = 0.99
    ) -> tuple[list[DNAstrand], list[DNAstrand]]:
        captured = []

        tube_copy = dna_pool.copy()

        for dna in tube_copy:
            if (target_sequence in dna.sequence or target_sequence[::-1] in dna.sequence) and dna.is_double is False:
                if random.random() < success_chance:
                    captured.append(dna)
                    dna_pool.remove(dna)

        return captured, dna_pool

    # Handle cutting into fragments of a single strand
    @staticmethod
    def cleave_single_stand(
            sequence: str,
            enzyme_site: str,
            sticky_length: int = 0
    ) -> List[DNAstrand]:
        fragments = sequence.split(enzyme_site)

        site_length = len(enzyme_site)

        modified_frags = []

        for index, frag in enumerate(fragments):
            if index == 0:
                modified_frags.append(frag + enzyme_site[0: site_length-sticky_length])
            elif index + 1 == len(fragments):
                modified_frags.append(enzyme_site[site_length-sticky_length: site_length] + frag)
            else:
                modified_frags.append(enzyme_site[site_length-sticky_length: site_length] + frag
                                      + enzyme_site[sticky_length: site_length])

        return [DNAstrand(frag) for frag in modified_frags]

    # Seperate strand at specific site using cleaving enzyme
    @staticmethod
    def cleave(
        dna: DNAstrand,
        enzyme_site: str,
        sticky_length: int = 0,
        p_fail: float = 0.02
    ) -> List[DNAstrand]:
        # Small change to fail
        if random.random() < p_fail:
            return [dna]

        fragments = []

        # Handle main strand first
        fragments.extend(VirtualLab.cleave_single_stand(dna.sequence, enzyme_site, sticky_length))

        if dna.is_double:
            # Add fragments from second strand
            fragments.extend(VirtualLab.cleave_single_stand(dna.upper_sequence, enzyme_site[::-1],
                                                            len(enzyme_site) - sticky_length))

        return fragments

    # Connect all possible fragments in the pool
    @staticmethod
    def ligation(
        fragments: List[DNAstrand],
        error_chance: float = 0.01
    ) -> List[DNAstrand]:
        while len(fragments) > 1:
            if random.random() < error_chance:
                break

            frag_copies = fragments.copy()

            random.shuffle(frag_copies)

            current_candidate = frag_copies[0]

            # For each possible fragment, if we can form a double helix with another create a new combined strand
            for frag in frag_copies[1:]:
                if current_candidate.can_form_double_helix(frag):
                    fragments.remove(frag)
                    current_candidate.sequence = current_candidate.sequence + frag.upper_sequence
                    current_candidate.modify_second_strand(current_candidate.upper_sequence + frag.sequence)

        return fragments


# Get the complement of a sequence
def reverse_complement(seq) -> str:
    complement = get_complement(seq)
    return complement[::-1]


# Generate random unique genetic codes (without overlap from neighbors)
def generate_variable_codes(n, length=6, seed=24):
    random.seed(seed)
    forbidden_start = 'T'
    delimiter = 'TT'

    codes = set()

    # Defined to make use of delimiter without redefining, checks if a sequence is legal to add from random generation
    def is_safe(sequence: str):
        if (delimiter in sequence) \
                or (sequence+delimiter in codes) \
                or (reverse_complement(sequence)+delimiter in codes) \
                or (sequence[::-1]+delimiter in codes) \
                or (sequence[0] == forbidden_start) \
                or (sequence[-1] == forbidden_start):
            return False
        return True

    bad_count = 0

    # Generate a new code and check if it can be added based on delimiter and existing codes
    while len(codes) < n and bad_count < len(bases)**length:
        seq = ''.join(random.choice(bases) for _ in range(length))
        if is_safe(seq):
            codes.add(seq+delimiter)
        else:
            bad_count += 1

    # If we finished prev loop and still need to generate more, try creating them at a longer length
    if len(codes) < n:
        return generate_variable_codes(n, length+1, seed)

    return list(codes)


# Code to simulate running a solution to 3-SAT using the virtual lab
def main():
    # The expression itself
    CNF_list = [[['a', '-a', 'b'], ['a', 'b', 'c'], ['a', '-b', 'c'], ['-a', '-c', 'b']],
                [['a', 'a', 'b']],
                [['a', 'a', 'a'], ['-a', '-a', '-a']]]
    for cnf_index, CNF in enumerate(CNF_list):
        arguments = []

        # Count all the unique arguments
        for expression in CNF:
            for argument in expression:
                if argument.replace('-', '') not in arguments:
                    arguments.append(argument.replace('-', ''))

        codes = generate_variable_codes((len(arguments) * 3 + 1), 3)

        arg_count = len(arguments)
        argument_dict = {}
        genetic_code_dict = {}

        index = 0

        # Create dict to translate argument to genetic sequence and vice-versa to compare in the expression checking loop
        for index, argument in enumerate(arguments):
            arg_code = codes[index]
            complement_arg_code = codes[index+arg_count]
            middle_path_arg_code = codes[index+arg_count*2]

            argument_dict[argument] = arg_code
            argument_dict['-'+argument] = complement_arg_code
            argument_dict['set_'+argument] = middle_path_arg_code

            genetic_code_dict[arg_code] = argument
            genetic_code_dict[complement_arg_code] = '-'+argument
            genetic_code_dict[middle_path_arg_code] = 'set_'+argument

        argument_dict['set_last'] = codes[index+arg_count*2+1]
        genetic_code_dict[codes[index+arg_count*2+1]] = 'set_last'

        print(argument_dict)
        print(genetic_code_dict)

        dna_strands = []

        # Generate all possible paths, 2^n possibilities
        for mask in range(2 ** arg_count):
            result = []
            for i in range(arg_count):
                result.append(argument_dict['set_'+arguments[i]])
                if mask & (1 << (arg_count - i - 1)):
                    result.append(argument_dict['-'+arguments[i]])
                else:
                    result.append(argument_dict[arguments[i]])

            result.append(argument_dict['set_last'])
            dna_strands.append(DNAstrand("".join(result)))

        # The algorithm given in the text book. for each expression check that all the generated codes satisfy it
        for expression in CNF:
            bank_tube = []
            for argument in expression:
                containing, non_containing = VirtualLab.extract(dna_strands, argument_dict[argument])
                bank_tube.extend(containing)
            dna_strands = bank_tube

        dna_strands = VirtualLab.amplify(dna_strands, argument_dict['set_'+arguments[0]], argument_dict['set_last'])

        if len(dna_strands) > 0:
            print('solutions found for problem number ' + str(cnf_index))
        else:
            print('no solutions found for problem number ' + str(cnf_index))
        # print(dna_strands)

    # # Checkign double helix working correctly
    # first = DNAstrand('AAATTGG')
    # second = DNAstrand('TTAACC')
    # print(first.can_form_double_helix(second))


if __name__ == "__main__":
    main()
