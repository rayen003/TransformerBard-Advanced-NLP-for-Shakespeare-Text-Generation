import tensorflow as tf
from transformers import AutoTokenizer
import os
import warnings
import urllib3
import numpy as np

# Filter warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=urllib3.exceptions.NotOpenSSLWarning)


class Preprocessing_pipeline:
    def __init__(self, file_path, max_length=32, batch_size=32, model_name='prajjwal1/bert-tiny'):
        self.file_path = file_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
    
    def prepare_data(self):
        """Prepare and return dataset"""
        print("\nPreparing data...")

        # Read text
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        print(f"Read {len(text)} characters")

        # Tokenize text
        print("\nTokenizing text...")
        tokens = self.tokenizer(text, return_tensors='tf', truncation=False)  # Don't truncate for training data
        token_ids = tokens['input_ids'].numpy()[0]
        print(f"Generated {len(token_ids)} tokens")

        # Create sequences
        max_sequences = len(token_ids) - self.max_length - 1  # -1 for target token
        
        print("\nGenerating sequences...")
        print(f"- Maximum sequences possible: {max_sequences}")
        print(f"- Sequence length: {self.max_length}")

        # Generate sequences
        input_sequences = []
        target_sequences = []

        for i in range(max_sequences):
            start_idx = i
            end_idx = start_idx + self.max_length
            
            # Get input sequence and next token
            sequence = token_ids[start_idx:end_idx]
            next_token = token_ids[end_idx]
            
            input_sequences.append(sequence)
            target_sequences.append(next_token)

        # Convert to numpy arrays
        input_sequences = np.array(input_sequences)  # Shape: [num_sequences, seq_length]
        target_sequences = np.array(target_sequences)  # Shape: [num_sequences]

        # Convert to TensorFlow tensors
        input_sequences = tf.convert_to_tensor(input_sequences, dtype=tf.int32)
        target_sequences = tf.convert_to_tensor(target_sequences, dtype=tf.int32)

        print("\nCreating dataset...")
        print(f"Input shape: {input_sequences.shape}")
        print(f"Target shape: {target_sequences.shape}")
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            input_sequences,
            target_sequences
        ))
        
        # Prepare dataset for training
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
        
        # No need to reshape targets, they're already in the right shape
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

    @property
    def vocabulary_size(self):
        """Get vocabulary size"""
        return self.vocab_size


if __name__ == '__main__':
    print("Testing preprocessing pipeline...")
    PATH = "/Users/rayengallas/Desktop/Coding_projects/Project/data/shakespeare.txt"
    
    # Initialize preprocessor
    preprocessor = Preprocessing_pipeline(
        file_path=PATH,
        max_length=32,
        batch_size=32,
        model_name='prajjwal1/bert-tiny'
    )
    
    # Create dataset
    dataset = preprocessor.prepare_data()
    
    print("\nPreprocessing Results:")
    print("=" * 20)
    print(f"Vocabulary Size: {preprocessor.vocabulary_size}")
    print(f"Batch Size: {preprocessor.batch_size}")
    
    # Print sample shapes
    print("\nChecking data shapes...")
    for inputs, targets in dataset.take(1):
        print("\nSample batch shapes:")
        print(f"Input shape: {inputs.shape}")  # Should be (batch_size, seq_len)
        print(f"Target shape: {targets.shape}")  # Should be (batch_size,)
        
        # Print value ranges
        print("\nValue ranges:")
        print(f"Inputs - min: {tf.reduce_min(inputs)}, max: {tf.reduce_max(inputs)}")
        print(f"Targets - min: {tf.reduce_min(targets)}, max: {tf.reduce_max(targets)}")
    print("=" * 20)