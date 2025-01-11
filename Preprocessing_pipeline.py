import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import to_categorical
import warnings
import random

warnings.filterwarnings('ignore')

class Preprocessing_pipeline:
    def __init__(self, file_path, sequence_length=5) -> None:
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.tokenizer = Tokenizer()
        self.vocab_size = None
        self.embedding_dim = 64
        self.embedding_layer = None
    
    def get_raw_text(self):
        with open(self.file_path, 'r') as file:
            return file.read()
    
    def tokenize(self, text):
        self.tokenizer.fit_on_texts([text])
        self.vocab_size = len(self.tokenizer.word_index) + 1
        tokenized_text = self.tokenizer.texts_to_sequences([text])[0]
        
        # Create embedding layer now that we know vocab_size
        self.embedding_layer = Embedding(input_dim=self.vocab_size,
                                      output_dim=self.embedding_dim)
        return tokenized_text
    
    def create_sequences(self, tokenized_text):
        sequences = []
        for i in range(len(tokenized_text) - self.sequence_length):
            sequences.append(tokenized_text[i:i + self.sequence_length + 1])
        return sequences
    
    def data_generator(self, sequences, batch_size):
        num_sequences = len(sequences)
        while True:  # Loop forever
            # Shuffle sequences at the start of each epoch
            indices = np.random.permutation(num_sequences)
            
            for start_idx in range(0, num_sequences, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_sequences = [sequences[i] for i in batch_indices]
                
                # Pad sequences in this batch
                padded_batch = pad_sequences(
                    batch_sequences,
                    maxlen=self.sequence_length + 1,
                    padding='pre',
                    truncating='pre'
                )
                
                # Split into X and y
                X_batch = padded_batch[:, :-1]  # All tokens except last
                y_batch = padded_batch[:, -1]   # Only last token
                
                # Convert to tensors and apply transformations
                X_batch = self.embedding_layer(X_batch)
                
                # Create one-hot targets for each position in sequence
                y_batch = tf.one_hot(y_batch, self.vocab_size)
                y_batch = tf.repeat(tf.expand_dims(y_batch, axis=1), self.sequence_length, axis=1)
                
                yield X_batch, y_batch
    
    def __call__(self, batch_size=32):
        raw_text = self.get_raw_text()
        tokenized_text = self.tokenize(raw_text)
        sequences = self.create_sequences(tokenized_text)
        
        # Return the generator and additional info
        return (
            self.data_generator(sequences, batch_size),
            self.vocab_size,
            len(sequences)  
        )


PATH = "/Users/rayengallas/Desktop/Coding_projects/Project/shakespeare.txt"
if __name__ == '__main__':
    print("Starting preprocessing...")
    preprocessor = Preprocessing_pipeline(PATH)
    generator, vocab_size, num_sequences = preprocessor(batch_size=32)
    
    print("\nPreprocessing Results:")
    print("-" * 20)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Total sequences: {num_sequences}")
    
    # Test the generator
    print("\nTesting generator...")
    X_batch, y_batch = next(generator)
    print(f"Batch input shape: {X_batch.shape}")
    print(f"Batch target shape: {y_batch.shape}")