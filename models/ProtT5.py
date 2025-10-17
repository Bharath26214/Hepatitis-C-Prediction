import os
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import numpy as np

# Force CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class ProtT5EmbedderCPU:
    def __init__(self, model_name="Rostlab/prot_t5_xl_bfd"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = TFAutoModel.from_pretrained(model_name)

    def embed_sequences(self, sequences, max_length=1024):
        embeddings = []
        for seq in sequences:
            # Preprocess sequence
            seq = " ".join(list(seq.replace("U", "X").replace("O", "X").replace("B", "X").replace("Z", "X")))
            inputs = self.tokenizer(seq, return_tensors="tf", padding="max_length",
                                    truncation=True, max_length=max_length)
            
            outputs = self.model(**inputs)
            # Take mean pooling over the token embeddings
            last_hidden_state = outputs.last_hidden_state
            mean_embedding = tf.reduce_mean(last_hidden_state, axis=1)
            embeddings.append(mean_embedding.numpy()[0])
        
        return np.array(embeddings)

if __name__ == "__main__":
    sequences = ["MEEPQSDPSV", "GAVLILLLV"]
    embedder = ProtT5EmbedderCPU()
    embeddings = embedder.embed_sequences(sequences)
    print("Embeddings shape:", embeddings.shape)
    print(embeddings)
