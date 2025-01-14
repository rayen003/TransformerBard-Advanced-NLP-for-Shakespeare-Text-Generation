import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

class Preprocessing_pipeline:
    def __init__(self, file_path, max_length=50, batch_size=16) -> None:
        self.file_path = file_path
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Initialize tiny BERT tokenizer (much smaller model)
        model_name = 'prajjwal1/bert-tiny'  # Only 4.4MB
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        
    def get_raw_text(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Error: Could not find file at {self.file_path}")
            raise
    
    def create_sequences(self, text, max_sequences=1000):
        """Create overlapping sequences from text"""
        # Tokenize the entire text first
        encoded = self.tokenizer(text, return_tensors='tf')
        tokens = encoded['input_ids'][0]  # Get the token IDs
        
        # Calculate total possible sequences
        total_sequences = len(tokens) - self.max_length
        max_sequences = min(total_sequences, max_sequences)
        
        # Initialize arrays for sequences
        all_input_sequences = []
        all_target_sequences = []
        all_attention_masks = []
        
        # Process in batches
        batch_size = 100
        for start_idx in range(0, max_sequences, batch_size):
            end_idx = min(start_idx + batch_size, max_sequences)
            batch_size_actual = end_idx - start_idx
            
            # Initialize tensors for this batch
            input_sequences = []
            target_sequences = []
            attention_masks = []
            
            # Create sequences for this batch
            for i in range(start_idx, end_idx):
                # Get input sequence
                input_seq = tokens[i:i + self.max_length]
                # Get target (next token)
                target = tokens[i + self.max_length]
                
                # Pad input sequence if needed
                if len(input_seq) < self.max_length:
                    padding = tf.zeros(self.max_length - len(input_seq), dtype=tf.int32)
                    input_seq = tf.concat([input_seq, padding], axis=0)
                
                input_sequences.append(input_seq)
                target_sequences.append(target)
                attention_masks.append(tf.ones(self.max_length))  # All tokens are real (no padding)
            
            # Stack sequences
            input_sequences = tf.stack(input_sequences)
            target_sequences = tf.one_hot(target_sequences, depth=self.vocab_size)
            attention_masks = tf.stack(attention_masks)
            
            # Append to main lists
            all_input_sequences.append(input_sequences)
            all_target_sequences.append(target_sequences)
            all_attention_masks.append(attention_masks)
        
        # Concatenate all batches
        return (
            tf.concat(all_input_sequences, axis=0),
            tf.concat(all_target_sequences, axis=0),
            tf.concat(all_attention_masks, axis=0)
        )
    
    def data_generator(self, input_sequences, target_sequences, attention_masks, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': input_sequences,
                'attention_mask': attention_masks,
            },
            target_sequences
        ))
        return dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)  # Reduced shuffle buffer
    
    def prepare_data(self, max_sequences=1000):
        """Prepare data for training"""
        # Get text and create sequences
        text = self.get_raw_text()
        self.input_sequences, self.target_sequences, self.attention_masks = self.create_sequences(text, max_sequences)
        
        return (self.input_sequences, self.target_sequences, self.attention_masks)
    
    def __call__(self, batch_size=16):
        self.prepare_data()
        generator = self.data_generator(self.input_sequences, self.target_sequences, self.attention_masks, batch_size)
        
        print("\nPreprocessing Results:")
        print("=" * 20)
        print(f"Vocabulary Size: {self.vocab_size}")
        print(f"Batch Size: {batch_size}")
        print("=" * 20)
        
        return generator, self.vocab_size


PATH = "/Users/rayengallas/Desktop/Coding_projects/Project/data/shakespeare.txt"
if __name__ == '__main__':
    preprocessor = Preprocessing_pipeline(PATH)
    preprocessor(batch_size=16)
