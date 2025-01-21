import sys
import os
import warnings
import urllib3
import tensorflow as tf
warnings.filterwarnings('ignore', category=urllib3.exceptions.NotOpenSSLWarning)

sys.path.append('/Users/rayengallas/Desktop/Coding_projects/Project')
from Preprocessing_pipeline import Preprocessing_pipeline

def main():
    print("Starting preprocessing test...")
    
    # Check file path
    file_path = "/Users/rayengallas/Desktop/Coding_projects/Project/data/shakespeare.txt"
    print(f"\nChecking file path: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    try:
        # Initialize preprocessor
        print("\nInitializing preprocessor...")
        preprocessor = Preprocessing_pipeline(
            file_path=file_path,
            max_length=50,
            batch_size=16
        )
        print("Preprocessor initialized successfully")
        print(f"Vocabulary size: {preprocessor.vocab_size}")
        
        # Get raw text
        print("\nReading text file...")
        text = preprocessor.get_raw_text()
        words = text.split()
        print(f"Text length: {len(text)} characters")
        print(f"Total words: {len(words)}")
        print(f"Maximum possible sequences: {len(words) - preprocessor.max_length}")
        print("\nFirst 100 characters:")
        print(text[:100])
        
        # Create sequences
        print("\nCreating sequences...")
        max_sequences = 500
        input_sequences, target_sequences, attention_masks = preprocessor.create_sequences(text, max_sequences)
        
        print("\nSequence Information:")
        print(f"Number of sequences: {len(input_sequences)}")
        print(f"Input sequence length: {input_sequences.shape[1]}")
        print(f"Target sequence shape: {target_sequences.shape}")  # Should be (num_sequences, vocab_size)
        
        # Show example of input and target
        print("\nExample of first sequence:")
        # Decode the input sequence
        first_input = input_sequences[0].numpy()
        input_text = preprocessor.tokenizer.decode(first_input)
        print(f"Input text: {input_text}")
        
        # Get the target token
        target_idx = tf.argmax(target_sequences[0]).numpy()
        target_token = preprocessor.tokenizer.decode([target_idx])
        print(f"Target token: {target_token}")
        
        # Test data generator
        print("\nTesting data generator...")
        dataset = preprocessor.data_generator(
            input_sequences[:50],
            target_sequences[:50],
            attention_masks[:50],
            batch_size=16
        )
        
        # Get first batch and show example
        for batch in dataset.take(1):
            inputs, targets = batch
            print("\nFirst batch information:")
            print(f"Input IDs shape: {inputs['input_ids'].shape}")
            print(f"Attention mask shape: {inputs['attention_mask'].shape}")
            print(f"Target shape: {targets.shape}")
            
            # Show first example in batch
            batch_input = inputs['input_ids'][0].numpy()
            batch_target = tf.argmax(targets[0]).numpy()
            print("\nFirst example in batch:")
            print(f"Input: {preprocessor.tokenizer.decode(batch_input)}")
            print(f"Target token: {preprocessor.tokenizer.decode([batch_target])}")
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
