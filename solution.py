import argparse

# using OR-Tools to solve the problem
from ortools.sat.python import cp_model
from abc import ABC, abstractmethod


def main(args):
    cap_name_set = read_dataset(args.name_dataset)
    alphabet_set = get_alphabet_set(cap_name_set)
    encoder_and_decoder = EncoderAndDecoder(alphabet_set, cap_name_set)

    # this is not an accurate upper bound. the max number of block types is {max_lenth}_C_{6}.
    # however, it makes the problem too hard to solve. so we use a smaller number.
    # in this problem, we have hard limit of 6 block types.
    max_num_block_type = 6

    nameblock_model = NameBlockModel(
        cap_name_set, alphabet_set, encoder_and_decoder, max_num_block_type
    )

    nameblock_model.solve_step_1()
    nameblock_model.solve_step_2()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-dataset", type=str, default="input.txt")
    return parser.parse_args()


def read_dataset(dataset_path):
    cap_name_set = []

    with open(dataset_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            cap_name_set.append(line.strip().upper())
    return cap_name_set


def get_alphabet_set(names):
    alphabet_set = set()
    for name in names:
        for letter in name:
            alphabet_set.add(letter)

    alphabet_set = sorted(list(alphabet_set))
    alphabet_set.append("*")

    if len(alphabet_set) < 6:
        raise Exception(
            "alphabet_set is too small. threre should be at least 6 letters because a block has 6 sides."
        )

    return alphabet_set


class EncoderAndDecoder:
    def __init__(self, alphabet_set, name_set):
        self.alphabet_set = alphabet_set
        self.max_length = max([len(name) for name in name_set])

    def encode_letter(self, letter):
        if letter not in self.alphabet_set:
            return "*"
        return self.alphabet_set.index(letter)

    def decode_letter(self, index):
        if index >= len(self.alphabet_set):
            return "*"
        return self.alphabet_set[index]

    def encode_name(self, name):
        encoded_name = []
        for letter in name:
            encoded_name.append(self.encode_letter(letter))

        # pad the encoded name with the placeholder
        while len(encoded_name) < self.max_length:
            encoded_name.append(self.encode_letter("*"))
        return encoded_name

    def decode_name(self, encoded_name):
        name = []
        for index in encoded_name:
            name.append(self.decode_letter(index))
        return name


class NameBlockModel(ABC):
    def __init__(self, name_set, alphabet_set, encoder_and_decoder, max_num_block_type):
        self.model = cp_model.CpModel()
        self.name_set = name_set
        self.alphabet_set = alphabet_set
        self.encoder_and_decoder = encoder_and_decoder
        self.max_num_block_type = max_num_block_type
        self.solver = cp_model.CpSolver()
        self.solver.parameters.enumerate_all_solutions = False
        self.solver.parameters.num_search_workers = 8

        self.num_names = len(name_set)
        self.num_alphabets = len(alphabet_set)
        self.max_name_length = max([len(name) for name in name_set])

    def solve_step_1(self):
        print("Solving step 1... it is going to take a while.")
        self._init_variables()
        self._init_step_1_constraints()
        self._init_step_1_optimzation_objective()

        status = self.solver.Solve(self.model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print("Solution found.")
            pass
        else:
            raise Exception("No solution found for step 1.")

    def solve_step_2(self):
        print("Solving step 2... it is going to take a while.")
        self._add_step_2_constraints()
        self._add_optimization_objective()

        status = self.solver.Solve(self.model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print("Solution found.")
            pass
        else:
            raise Exception("No solution found for step 2.")

        solution = self.solution_to_string()
        print(solution)
        with open("output.txt", "w") as f:
            f.write(solution)

    def _init_variables(self):
        # P: mapping from name to the index of a block. P[i][j] = k means the jth letter of the ith name is in the kth type block
        # P is a 2D array
        P = []
        for i in range(self.num_names):
            P.append([])
            for j in range(self.max_name_length):
                P[i].append(
                    self.model.NewIntVar(
                        0, self.max_num_block_type - 1, "P[%i,%i]" % (i + 1, j + 1)
                    )
                )

        # N: represents the name set. N[i][j] = a means the jth letter of the ith name has ath letter in the alphabet.
        # N is a 2D array

        N = []
        for name in self.name_set:
            N.append(self.encoder_and_decoder.encode_name(name))

        # B: represents what letters are on what types of the blocks. B[i][j] = 1/0 means the ith alphabet is in the jth type block.
        # using letter as index seems unintuitive, but it is necessary becuase a letter
        B = []
        for i in range(self.num_alphabets):
            B.append([])
            for j in range(self.max_num_block_type):
                B[i].append(self.model.NewIntVar(0, 1, "B[%i,%i]" % (i, j)))

        # C: represents the number of blocks of each type. C[i] = k means there are k blocks of the ith type.
        # if the ith type block is not used, then C[i] = 0
        # the upper bound is max_length because having the same block type for each letter is the worst case.
        C = []
        for i in range(self.max_num_block_type):
            C.append(self.model.NewIntVar(0, self.max_name_length, "C[%i]" % i))

        # U: represents whether a ith type block is used. U[i] = True/False means the ith type block is used/not used.
        U = []
        for i in range(self.max_num_block_type):
            U.append(self.model.NewBoolVar("E[%i]" % i))

        self.P = P
        self.N = N
        self.B = B
        self.C = C
        self.U = U

    def _init_step_1_constraints(self):
        # constraint 1: if a letter in a name is a block, then the block must have the letter
        for i in range(self.num_names):
            for j in range(self.max_name_length):
                letter = self.N[i][j]
                block = self.P[i][j]
                self.model.AddElement(block, self.B[letter], 1)

        # constraint 2: the number of blocks of each type in a name must be equal or less than the total number of blocks of the type
        # it is not possible to count the IntVars that satisfies a certain condition.
        # therefore, we need to create a new variable that is True if the condition is satisfied, and False otherwise.
        # and add them up to get the count.
        # in otherwards, rather than assigning a value to the variable (which is not possible), create a constraint that the variable must be equal to the value.
        for i in range(self.num_names):
            for j in range(self.max_num_block_type):
                block_type_counters = []
                for k in range(self.max_name_length):
                    block_type_counter = self.model.NewBoolVar(
                        "block type %i is used in %ith letter of %ith name" % (j, k, i)
                    )

                    # this looks like the opposite of the intuition, but it is correct.
                    # https://groups.google.com/g/or-tools-discuss/c/xpo1JdPzh
                    self.model.Add(self.P[i][k] == j).OnlyEnforceIf(block_type_counter)
                    self.model.Add(self.P[i][k] != j).OnlyEnforceIf(
                        block_type_counter.Not()
                    )
                    block_type_counters.append(block_type_counter)

                self.model.Add(sum(block_type_counters) <= self.C[j])

        # constraint 3: one block must have six letters
        num_faces = 6
        for j in range(self.max_num_block_type):
            self.model.Add(
                sum([self.B[i][j] for i in range(self.num_alphabets)]) == num_faces
            )

    def _init_step_1_optimzation_objective(self):
        # optimize 1: minimize the total number of block types
        for i in range(self.max_num_block_type):
            block_type_exist = self.U[i]
            self.model.Add(self.C[i] > 0).OnlyEnforceIf(block_type_exist)
            self.model.Add(self.C[i] == 0).OnlyEnforceIf(block_type_exist.Not())

            self.U.append(block_type_exist)

        self.model.Minimize(sum(self.U))

    def _add_step_2_constraints(self):
        ## constraint P
        for i in range(self.num_names):
            for j in range(self.max_name_length):
                self.model.Add(self.P[i][j] == self.solver.Value(self.P[i][j]))

        ## constraint B
        for i in range(self.num_alphabets):
            for j in range(self.max_num_block_type):
                self.model.Add(self.B[i][j] == self.solver.Value(self.B[i][j]))

        ## constraint E
        for i in range(self.max_num_block_type):
            self.model.Add(self.U[i] == self.solver.Value(self.U[i]))

        ## add hints for C.
        ## we do not constraint C, because this is what we want to optimize.
        for i in range(self.max_num_block_type):
            self.model.AddHint(self.C[i], self.solver.Value(self.C[i]))

        # constraint: make sure that C[i] remains 0 if the block type is not used in the first optimization
        for i in range(self.max_num_block_type):
            self.model.Add(self.C[i] == 0).OnlyEnforceIf(
                not self.solver.BooleanValue(self.U[i])
            )
            self.model.Add(self.C[i] > 0).OnlyEnforceIf(
                self.solver.BooleanValue(self.U[i])
            )

    def _add_optimization_objective(self):
        # optimization 2: minimize the difference between the number of blocks of each type, in other words, min the max of the number of blocks of each type.
        max_C = self.model.NewIntVar(0, self.max_name_length, "max_count")
        self.model.AddMaxEquality(max_C, self.C)

        non_zero_min_C = self.model.NewIntVar(0, self.max_name_length, "min_count")
        self.model.AddMinEquality(
            non_zero_min_C,
            [
                self.C[i]
                for i in range(self.max_num_block_type)
                if self.solver.Value(self.U[i]) == 1
            ],
        )

        self.model.Minimize((max_C - non_zero_min_C) + sum(self.C))

    def solution_to_string(
        self,
    ):
        result = ""
        P = self.P
        N = self.N
        B = self.B
        C = self.C

        for j in range(self.max_num_block_type):
            num_blocks = self.solver.Value(C[j])
            result += f"block {j} ({num_blocks}):"

            for i in range(self.num_alphabets):
                if self.solver.Value(B[i][j]) == 1:
                    result += self.encoder_and_decoder.decode_letter(i) + " "
            result += "\n"

        for i in range(len(self.name_set)):
            name_in_fixed_length = ""
            for j in range(self.max_name_length):
                name_in_fixed_length += self.encoder_and_decoder.decode_letter(
                    self.solver.Value(N[i][j])
                )

            result += name_in_fixed_length + "\n"
            for j in range(self.max_name_length):
                result += str(self.solver.Value(P[i][j]))
            result += "\n"

        return result


if __name__ == "__main__":
    args = parse_args()
    main(args)
