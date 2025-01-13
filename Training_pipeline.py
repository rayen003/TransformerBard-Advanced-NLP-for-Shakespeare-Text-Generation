import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
import sys
sys.path.append('/Users/rayengallas/Desktop/Coding_projects/Project')
from preprocessing_pipeline import Preprocessing_pipeline
from model import TransformerModel
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import os
from tqdm.auto import tqdm

# Get preprocessed data with generator
def create_small_dataset(sequences, split_ratio=0.1, train_test_ratio=0.8):
    """
    Create a small dataset with only split_ratio% of the data
    Args:
        sequences: List of all sequences
        split_ratio: Percentage of total data to use (default 10%)
        train_test_ratio: Ratio of training to total data (default 80% train, 20% test)
    """
    # First reduce the dataset size
    num_sequences = len(sequences)
    small_size = int(num_sequences * split_ratio)
    small_dataset = sequences[:small_size]
    
    # Then split into train/test
    train_size = int(small_size * train_test_ratio)
    train_sequences = small_dataset[:train_size]
    test_sequences = small_dataset[train_size:]
    
    return train_sequences, test_sequences

# Get preprocessed data
print("Starting preprocessing...")
preprocessor = Preprocessing_pipeline(file_path="/Users/rayengallas/Desktop/Coding_projects/Project/shakespeare.txt")
raw_text = preprocessor.get_raw_text()
tokenized_text = preprocessor.tokenize(raw_text)
sequences = preprocessor.create_sequences(tokenized_text)

train_sequences, test_sequences = create_small_dataset(sequences, split_ratio=0.1)

print("\nDataset sizes:")
print(f"Total sequences: {len(sequences)}")
print(f"Training sequences (10%): {len(train_sequences)}")
print(f"Testing sequences (10%): {len(test_sequences)}")

# Model parameters
BATCH_SIZE = 32
EMBEDDING_DIM = 256  # Increased from 64
NUM_HEADS = 8
FF_DIM = 256  # Increased from 32
NUM_TRANSFORMER_BLOCKS = 6  # Increased from 4
DROPOUT = 0.1
MAX_SEQUENCE_LENGTH = 100

# Create model
print("\nStarting training...")
model = TransformerModel(
    vocab_size=preprocessor.vocab_size,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    dropout=DROPOUT,
    max_sequence_length=MAX_SEQUENCE_LENGTH
)

class Training_pipeline:
    def __init__(self, model, train_generator, test_generator, num_sequences, batch_size=32):
        self.model = model
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.batch_size = batch_size
        self.num_sequences = num_sequences
        self.steps_per_epoch = num_sequences // batch_size
        self.experiment_dir = "experiments"
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "weights"), exist_ok=True)
        
    def compile(self):
        # Create learning rate schedule
        initial_learning_rate = 1e-4  # Reduced initial learning rate
        decay_steps = 1000
        decay_rate = 0.98

        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps,
            decay_rate,
            staircase=True
        )
        
        # Use AdamW optimizer with weight decay and gradient clipping
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate_schedule,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0  # Added gradient clipping
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
    
    def save_experiment_results(self, history, test_results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save experiment results
        results_filename = os.path.join(self.experiment_dir, f"experiment_{timestamp}.json")
        
        # Convert history to a dictionary with lists
        history_dict = {}
        for key in history.history.keys():
            history_dict[key] = [float(val) for val in history.history[key]]
        
        results = {
            "timestamp": timestamp,
            "config": {
                "batch_size": self.batch_size,
                "vocab_size": preprocessor.vocab_size,
                "embedding_dim": EMBEDDING_DIM,
                "num_heads": NUM_HEADS,
                "sequences": {
                    "total": len(sequences),
                    "train": len(train_sequences),
                    "test": len(test_sequences)
                },
                "tokenizer": {
                    "type": "Custom Keras Tokenizer",
                    "vocab_size": preprocessor.vocab_size,
                    "char_level": False,
                    "filters": ""
                },
                "embedding": {
                    "type": "Custom trainable embeddings",
                    "dimension": EMBEDDING_DIM,
                    "trainable": True,
                    "initialization": "random uniform"
                }
            },
            "history": history_dict,
            "test_results": {
                "loss": float(test_results[0]),
                "accuracy": float(test_results[1])
            }
        }
        
        with open(results_filename, "w") as f:
            json.dump(results, f, indent=4)
        
        # Save model weights
        weights_path = os.path.join(self.experiment_dir, "weights", f"weights_{timestamp}.weights.h5")
        self.model.save_weights(weights_path)
        
        return results_filename, weights_path
    
    def train(self, epochs=10):
        print("\nStarting training...")
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.test_generator,
            validation_steps=self.steps_per_epoch,
            verbose=1 
        )
        
        print("\nEvaluating on test set...")
        test_results = self.model.evaluate(
            self.test_generator,
            steps=self.steps_per_epoch,
            verbose=2  # Show more detailed progress
        )
        
        print("\nTest Results:")
        print(f"Loss: {test_results[0]:.4f}")
        print(f"Accuracy: {test_results[1]:.4f}")
        
        return history
    
    def evaluate(self, test_generator, test_steps):
        return self.model.evaluate(
            test_generator,
            steps=test_steps
        )
    
    def __call__(self, epochs=10):
        self.compile()
        print("\nStarting training...")
        history = self.train(epochs)
        results_path, weights_path = self.save_experiment_results(history, self.evaluate(self.test_generator, self.steps_per_epoch))
        print(f"\nExperiment results saved to: {results_path}")
        print(f"Model weights saved to: {weights_path}")
        return history


if __name__ == '__main__':
    # Create generators
    train_generator = preprocessor.data_generator(train_sequences, BATCH_SIZE)
    test_generator = preprocessor.data_generator(test_sequences, BATCH_SIZE)
    
    # Create training pipeline
    pipeline = Training_pipeline(
        model=model,
        train_generator=train_generator,
        test_generator=test_generator,
        num_sequences=len(train_sequences),
        batch_size=BATCH_SIZE
    )
    
    # Train the model
    pipeline(epochs=10)