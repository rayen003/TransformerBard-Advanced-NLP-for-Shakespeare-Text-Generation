import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
import os
import warnings
import urllib3

warnings.filterwarnings('ignore', category=urllib3.exceptions.NotOpenSSLWarning)


class Preprocessing_pipeline:
    def __init__(self, file_path, max_length=50, batch_size=32, model_name='prajjwal1/bert-tiny', use_percentage=1.0):
        self.file_path = file_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.use_percentage = use_percentage
    
    def get_raw_text(self):
        """Read raw text from file"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def create_sequences(self, tokens, max_sequences=None):
        """Create input sequences and targets with BERT embeddings"""
        # Calculate total possible sequences
        total_sequences = len(tokens) - self.max_length
        if max_sequences:
            total_sequences = min(total_sequences, max_sequences)
            
        # Initialize BERT model for embeddings
        bert_model = TFAutoModel.from_pretrained(self.model_name, from_pt=True)
        
        sequences = []
        next_tokens = []
        
        print(f"\nGenerating embeddings for {total_sequences} sequences...")
        
        # Create sequences using sliding window
        for i in range(total_sequences):
            if i % 100 == 0:  # Progress update more frequently
                print(f"Processing sequence {i}/{total_sequences}")
                
            # Get sequence and target
            sequence = tokens[i:i + self.max_length]
            next_token = tokens[i + self.max_length]
            
            # Convert to tensor and get BERT embedding
            inputs = {
                'input_ids': tf.convert_to_tensor([sequence]),
                'attention_mask': tf.convert_to_tensor([[1] * self.max_length])
            }
            
            # Get embeddings from BERT's last hidden state
            embeddings = bert_model(inputs).last_hidden_state[0]  # Shape: (seq_len, hidden_size)
            
            sequences.append(embeddings)
            next_tokens.append(tf.one_hot(next_token, self.vocab_size))
        
        print("Stacking sequences...")
        return (
            tf.stack(sequences),  # Shape: (num_sequences, seq_len, hidden_size)
            tf.stack(next_tokens)  # Shape: (num_sequences, vocab_size)
        )
    
    def create_dataset(self, inputs, targets, batch_size):
        """Create tf.data.Dataset from tensors"""
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        dataset = dataset.cache()  # Cache the dataset in memory
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Enable prefetching
        return dataset
    
    def prepare_data(self):
        """Prepare and return dataset using the class's use_percentage"""
        print("\nPreparing data with {}% of data...".format(self.use_percentage * 100))

        # Read and preprocess text
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        print("Text preparation:")
        print(f"- Total characters: {len(text)}")
        
        # Calculate how many characters to use based on percentage
        chars_to_use = int(len(text) * self.use_percentage)
        text = text[:chars_to_use]
        print(f"- Using {chars_to_use} characters ({self.use_percentage*100}% of total)")

        # Tokenize text
        print("\nTokenizing text...")
        tokens = self.tokenizer(text, return_tensors='tf', truncation=True)
        token_ids = tokens['input_ids'].numpy()[0]
        print(f"- Generated {len(token_ids)} tokens")

        # Create sequences
        max_sequences = len(token_ids) - self.max_length
        num_sequences = int(max_sequences * self.use_percentage)
        if num_sequences < 1:
            num_sequences = 1

        print("\nSequence preparation:")
        print(f"- Maximum possible sequences: {max_sequences}")
        print(f"- Using {num_sequences} sequences ({self.use_percentage*100}% of total)")
        print(f"- Each sequence length: {self.max_length} tokens")

        # Generate sequences
        input_sequences = []
        target_sequences = []

        for i in range(num_sequences):
            start_idx = i
            end_idx = start_idx + self.max_length
            sequence = token_ids[start_idx:end_idx]
            target = token_ids[start_idx + 1:end_idx + 1]
            
            input_sequences.append(sequence)
            target_sequences.append(target)

        # Convert to TensorFlow tensors
        input_sequences = tf.convert_to_tensor(input_sequences)
        target_sequences = tf.convert_to_tensor(target_sequences)

        print("\nCreating final dataset...")
        # Create and return the dataset
        return self.create_dataset(input_sequences, target_sequences, self.batch_size)
    
    @property
    def vocabulary_size(self):
        """Get vocabulary size"""
        return self.vocab_size


if __name__ == '__main__':
    import tensorflow as tf
    
    print("Testing preprocessing pipeline...")
    PATH = "/Users/rayengallas/Desktop/Coding_projects/Project/data/shakespeare.txt"
    
    # Initialize preprocessor with 20% of data
    preprocessor = Preprocessing_pipeline(
        file_path=PATH,
        max_length=50,
        batch_size=32,
        model_name='prajjwal1/bert-tiny',
        use_percentage=0.2  # 20%
    )
    
    print(f"\nStarting preprocessing with {preprocessor.use_percentage*100}% of data...")
    dataset = preprocessor.prepare_data()
    
    print("\nPreprocessing Results:")
    print("=" * 20)
    print(f"Vocabulary Size: {preprocessor.vocabulary_size}")
    print(f"Batch Size: {preprocessor.batch_size}")
    
    # Print sample shapes
    print("\nChecking data shapes...")
    for embeddings, targets in dataset.take(1):
        print("\nSample batch shapes:")
        print(f"BERT Embeddings shape: {embeddings.shape}")  # Should be (batch_size, seq_len, 768)
        print(f"Target shape: {targets.shape}")  # Should be (batch_size, vocab_size)
        
        # Print value ranges
        print("\nValue ranges:")
        print(f"Embeddings - min: {tf.reduce_min(embeddings):.4f}, max: {tf.reduce_max(embeddings):.4f}")
        print(f"Targets - min: {tf.reduce_min(targets):.4f}, max: {tf.reduce_max(targets):.4f}")
    print("=" * 20)
