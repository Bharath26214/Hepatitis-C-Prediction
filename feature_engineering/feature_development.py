import numpy as np
from itertools import product

class FeatureDevelopment:
  def __init__(self, seq):
    self.seq = seq
    self.length = len(seq)
    self.valid_aa = 'AMVFIHYLWCDKSTGQEPRN'
    self.hydrophobicity = {
        'A': 0.62, 'C': 0.29, 'D': -0.90, 'E': -0.74, 'F': 1.19,
        'G': 0.48, 'H': -0.40, 'I': 1.38, 'K': -1.50, 'L': 1.06,
        'M': 0.64, 'N': -0.78, 'P': 0.12, 'Q': -0.85, 'R': -2.53,
        'S': -0.18, 'T': -0.05, 'V': 1.08, 'W': 0.81, 'Y': 0.26
    }
    self.hydrophilicity = {
        'A': -0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C': -1.0,
        'Q': 0.2, 'E': 3.0, 'G': 0.0, 'H': -0.5, 'I': -1.8,
        'L': -1.8, 'K': 3.0, 'M': -1.3, 'F': -2.5, 'P': 0.0,
        'S': 0.3, 'T': -0.4, 'W': -3.4, 'Y': -2.3, 'V': -1.5
    }
    self.aa_mass = {
        'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15,
        'Q': 146.15, 'E': 147.13, 'G': 75.07, 'H': 155.16, 'I': 131.17,
        'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
        'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
    }
    self.molecular_volume = {
        'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1,'C': 108.5,
        'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7,
        'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
        'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0
    }

  # Local Sequence Encodings
  def AAC(self):
    aa_count = np.array([self.seq.count(aa) for aa in self.valid_aa], dtype=float)
    AAC_vector = aa_count / self.length

    return AAC_vector

  def PAAC(self, lam=3, w=0.05):
        """Deterministic PAAC"""
        aac_freq = self.AAC()
        L = self.length
        theta = []

        for l in range(1, lam + 1):
            val = 0.0
            for i in range(L - l):
                aa1, aa2 = self.seq[i], self.seq[i + l]
                if aa1 in self.hydrophobicity and aa2 in self.hydrophobicity:
                    val += (self.hydrophobicity[aa1] - self.hydrophobicity[aa2]) ** 2
            val /= (L - l)
            theta.append(val)

        denom = 1 + w * sum(theta)
        return np.concatenate([aac_freq / denom, (w * np.array(theta)) / denom])

  def APAAC(self, lam=3, w=0.05):
      """Deterministic APAAC"""
      aa_freq = self.AAC()
      L = self.length
      theta = []

      for l in range(1, lam + 1):
          corr = 0.0
          for i in range(L - l):
              aa1, aa2 = self.seq[i], self.seq[i + l]
              if (aa1 in self.hydrophobicity and aa2 in self.hydrophobicity and
                  aa1 in self.hydrophilicity and aa2 in self.hydrophilicity):
                  h1, h2 = self.hydrophobicity[aa1], self.hydrophobicity[aa2]
                  p1, p2 = self.hydrophilicity[aa1], self.hydrophilicity[aa2]
                  corr += ((h1 - h2) ** 2 + (p1 - p2) ** 2) / 2
          corr /= (L - l)
          theta.append(corr)

      denom = 1 + w * sum(theta)
      return np.concatenate([aa_freq / denom, (w * np.array(theta)) / denom])

  # Global Sequence Encodings
  def DPC(self):
    dipeptides = [''.join(p) for p in product(list(self.valid_aa), repeat=2)]
    counts = dict.fromkeys(dipeptides, 0)

    for i in range(self.length - 1):
        dipep = self.seq[i] + self.seq[i+1]
        if dipep in counts:
            counts[dipep] += 1

    DPC_vector = np.array([counts[dp] / (self.length - 1) for dp in dipeptides])
    return DPC_vector

  def DDC(self):
    aa_freq = {aa: self.seq.count(aa) / self.length for aa in self.valid_aa}

    observed = self.DPC()
    dipeptides = [''.join(p) for p in product(list(self.valid_aa), repeat=2)]

    expected = [aa_freq[dp[0]] * aa_freq[dp[1]] for dp in dipeptides]

    ddc_vector = observed - np.array(expected)
    return ddc_vector

  def TPC(self):
    tripeptides = [''.join(p) for p in product(list(self.valid_aa), repeat=3)]
    counts = dict.fromkeys(tripeptides, 0)

    for i in range(self.length - 2):
        tripep = self.seq[i] + self.seq[i+1] + self.seq[i+2]
        if tripep in counts:
            counts[tripep] += 1

    TPC_vector = np.array([counts[dp] / (self.length - 1) for dp in tripeptides])
    return TPC_vector

  # Physiochemical Properties
  def PCP(self):
    props = [self.hydrophobicity, self.molecular_volume, self.aa_mass]
    PCP_vector = []

    for prop in props:
        values = np.array([prop[aa] * self.seq.count(aa) for aa in self.valid_aa], dtype=float)
        values /= self.length
        PCP_vector.extend(values)

    return np.array(PCP_vector, dtype=float)

  def AAI(self):
    pass

  # K-mer
  def CKSAAP(self):
    CKSAAP_vector = []

    for k in range(4):
        pairs = [''.join(pair) for pair in product(self.valid_aa, repeat=2)]
        pairs = {pair: 0 for pair in pairs}
        for i in range(len(self.seq)):
            for j in range(i + k + 1, self.length):
                if j - i == k + 1:
                    pair = self.seq[i] + self.seq[j]
                    if set(pair).issubset(set(self.valid_aa)):
                        if pair in pairs:
                            pairs[pair] += 1
                        else:
                            pairs[pair] = 1
        if k == 0:
            for l in pairs:
                pairs[l] = pairs[l] / (self.length - 1)
        elif k == 1:
            for l in pairs:
                pairs[l] = pairs[l] / (self.length - 2)
        elif k == 2:
            for l in pairs:
                pairs[l] = pairs[l] / (self.length - 3)
        elif k == 3:
            for l in pairs:
                pairs[l] = pairs[l] / (self.length - 4)

        CKSAAP_vector.extend([pairs[pair] for _ in range(1) for pair in pairs])

    return np.array(CKSAAP_vector)