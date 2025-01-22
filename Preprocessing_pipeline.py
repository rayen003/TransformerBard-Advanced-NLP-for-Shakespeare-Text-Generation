import tensorflow as tf
from transformers import AutoTokenizer
import os
import warnings
import urllib3
import numpy as np

warnings.filterwarnings('ignore', category=urllib3.exceptions.NotOpenSSLWarning)


class Preprocessing_pipeline:
    def __init__(self, file_path, max_length=50, batch_size=32, model_name='prajjwal1/bert-tiny', use_percentage=0.01):
        self.file_path = file_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_name = model_name
        self.use_percentage = use_percentage
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

        # Create sequences with overlap for more training data
        max_sequences = len(token_ids) - self.max_length - 1  # -1 for target token
        
        # Calculate stride (how many tokens to skip between sequences)
        stride = self.max_length  # No overlap for faster processing
        
        # Calculate number of sequences with stride
        num_sequences = (max_sequences - 1) // stride + 1
        
        # Apply use_percentage to get final number of sequences
        num_sequences = int(num_sequences * self.use_percentage)
        
        print("\nDebug info:")
        print(f"- Token IDs length: {len(token_ids)}")
        print(f"- Max length: {self.max_length}")
        print(f"- Stride length: {stride}")
        print(f"- Max possible sequences with stride: {(max_sequences - 1) // stride + 1}")
        print(f"- Use percentage: {self.use_percentage}")
        print(f"- Number of sequences to generate: {num_sequences}")

        print("\nGenerating sequences...")
        
        # Pre-allocate numpy arrays for speed
        input_sequences = np.zeros((num_sequences, self.max_length), dtype=np.int32)
        target_sequences = np.zeros(num_sequences, dtype=np.int32)

        print("\nStarting sequence generation loop...")
        for i in range(num_sequences):
            if i % 1000 == 0:  # Print progress every 1000 sequences
                print(f"Generated {i}/{num_sequences} sequences")
                
            # Calculate start index using stride
            start_idx = i * stride
            end_idx = start_idx + self.max_length
            
            # Verify indices
            if end_idx + 1 > len(token_ids):
                print(f"Warning: End index {end_idx + 1} exceeds token_ids length {len(token_ids)}")
                break
                
            # Get input sequence and next token
            input_sequences[i] = token_ids[start_idx:end_idx]
            target_sequences[i] = token_ids[end_idx]

        print(f"\nFinished generating sequences:")
        print(f"- Actually generated: {len(input_sequences)} sequences")

        if len(input_sequences) == 0:
            raise ValueError("No sequences were generated! Check the use_percentage and max_length parameters.")

        # Convert to TensorFlow tensors directly from numpy arrays
        input_sequences = tf.convert_to_tensor(input_sequences, dtype=tf.int32)
        target_sequences = tf.convert_to_tensor(target_sequences, dtype=tf.int32)

        print("\nCreating dataset...")
        print(f"Input shape: {input_sequences.shape}")
        print(f"Target shape: {target_sequences.shape}")
        
        # Create dataset with optimized settings
        dataset = tf.data.Dataset.from_tensor_slices((
            input_sequences,
            target_sequences
        ))
        
        # Optimize dataset pipeline
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=1000)  # Reduced buffer size
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
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
        max_length=50,
        batch_size=32,
        model_name='prajjwal1/bert-tiny',
        use_percentage=1.0
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
        print(f"Input shape: {inputs.shape}")  #(batch_size, seq_len)
        print(f"Target shape: {targets.shape}")  #(batch_size,)
        
        # Print value ranges
        print("\nValue ranges:")
        print(f"Inputs - min: {tf.reduce_min(inputs)}, max: {tf.reduce_max(inputs)}")
        print(f"Targets - min: {tf.reduce_min(targets)}, max: {tf.reduce_max(targets)}")
    print("=" * 20)