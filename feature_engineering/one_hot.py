import numpy as np

class OneHotEncoder:
    def __init__(self, alphabet="ACDEFGHIKLMNPQRSTVWY"):
        self.alphabet = alphabet
        self.aa_to_index = {aa: i for i, aa in enumerate(alphabet)}
        self.num_aa = len(alphabet)

    def encode_sequence(self, sequence):
        seq_len = len(sequence)
        one_hot = np.zeros((seq_len, self.num_aa), dtype=np.float32)

        for i, aa in enumerate(sequence):
            idx = self.aa_to_index.get(aa, None)
            if idx is not None:
                one_hot[i, idx] = 1.0
        return one_hot

    def encode_many(self, sequences):
        max_len = max(len(seq) for seq in sequences)
        batch = np.zeros((len(sequences), max_len, self.num_aa), dtype=np.float32)

        for i, seq in enumerate(sequences):
            one_hot_seq = self.encode_sequence(seq)
            batch[i, :len(seq), :] = one_hot_seq

        return batch


