class Kmer:
    def __init__(self, data, k=3):
        self.data = data
        self.k = k
        self.kmer_features = []
        
    def encode_features(self):
        self.kmer_features = []
        for _, row in self.data.iterrows():  
            seq = row["epitope_name"] 
            kmers = [seq[j:j+self.k] for j in range(len(seq) - self.k + 1)]
            
            self.kmer_features.append(kmers)
        return self.kmer_features
            
        
        