import tensorflow as tf
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

class Preprocessing_pipeline:
    def __init__(self, file_path, max_length=50, batch_size=32, model_name='prajjwal1/bert-tiny'):
        self.file_path = file_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
    
    def get_raw_text(self):
        """Read raw text from file"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def create_sequences(self, text, max_sequences=None):
        """Create input sequences and targets"""
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        # Create sequences
        sequences = []
        attention_masks = []
        next_tokens = []
        
        for i in range(0, len(tokens) - self.max_length):
            if max_sequences and i >= max_sequences:
                break
                
            # Get sequence and target
            sequence = tokens[i:i + self.max_length]
            next_token = tokens[i + self.max_length]
            
            # Create attention mask
            attention_mask = [1] * self.max_length
            
            # Convert target to one-hot
            target = tf.one_hot(next_token, self.vocab_size)
            
            sequences.append(sequence)
            attention_masks.append(attention_mask)
            next_tokens.append(target)
        
        # Convert to tensors
        input_sequences = tf.convert_to_tensor(sequences)
        attention_masks = tf.convert_to_tensor(attention_masks)
        target_sequences = tf.stack(next_tokens)
        
        return input_sequences, target_sequences, attention_masks
    
    def prepare_training_data(self, use_percentage=0.15):
        """Prepare data for training using specified percentage"""
        # Get text and create sequences
        text = self.get_raw_text()
        total_sequences = len(text.split()) - self.max_length
        max_sequences = int(total_sequences * use_percentage)
        print(f"Using {max_sequences} sequences ({use_percentage*100}% of total)")
        
        # Get sequences
        input_sequences, target_sequences, attention_masks = self.create_sequences(
            text, max_sequences=max_sequences
        )
        
        # Convert to numpy for splitting
        input_sequences = input_sequences.numpy()
        target_sequences = target_sequences.numpy()
        attention_masks = attention_masks.numpy()
        
        # Split into train and validation
        train_inputs, val_inputs, train_targets, val_targets, train_masks, val_masks = train_test_split(
            input_sequences, target_sequences, attention_masks,
            test_size=0.1, random_state=42
        )
        
        # Create datasets
        train_data = {
            'input_ids': tf.convert_to_tensor(train_inputs),
            'attention_mask': tf.convert_to_tensor(train_masks)
        }
        val_data = {
            'input_ids': tf.convert_to_tensor(val_inputs),
            'attention_mask': tf.convert_to_tensor(val_masks)
        }
        
        # Create tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_data, tf.convert_to_tensor(train_targets))
        ).shuffle(1000).batch(self.batch_size)
        
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (val_data, tf.convert_to_tensor(val_targets))
        ).batch(self.batch_size)
        
        return train_dataset, val_dataset

PATH = "/Users/rayengallas/Desktop/Coding_projects/Project/data/shakespeare.txt"
if __name__ == '__main__':
    preprocessor = Preprocessing_pipeline(PATH)
    train_dataset, val_dataset = preprocessor.prepare_training_data()
    print("\nPreprocessing Results:")
    print("=" * 20)
    print(f"Vocabulary Size: {preprocessor.vocab_size}")
    print(f"Batch Size: {preprocessor.batch_size}")
    print("=" * 20)
