import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Attention, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Sequential
import sys
sys.path.append('/Users/rayengallas/Desktop/Coding_projects/Project')
from preprocessing_pipeline import Preprocessing_pipeline
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import os

# Scaled dot product attention function
def scaled_dot_product_attention(query, key, value, mask=None):
    # Compute the dot product of Q and K, transpose K using `transpose_b=True`
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    
    # Scaling factor for the dot product
    scaling_factor = tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))
    scaled_attention_scores = matmul_qk / scaling_factor
    
    # Apply mask if provided (this is to prevent attention to padding tokens)
    if mask is not None:
        scaled_attention_scores += (mask * -1e9)
    
    # Calculate attention weights using softmax
    attention_weights = tf.nn.softmax(scaled_attention_scores, axis=-1)
    
    # Multiply the attention weights by the value matrix to get the output
    output = tf.matmul(attention_weights, value)

    return output


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        
        self.depth = d_model // self.num_heads
        
        # Dense layers for generating Q, K, V
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size, sequence_length):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, sequence_length, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, depth)
    
    def concat_heads(self, x, batch_size, sequence_length):
        """Combine the heads back into d_model dimension."""
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, depth)
        return tf.reshape(x, (batch_size, sequence_length, self.d_model))
    
    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]
        sequence_length = tf.shape(query)[1]
        
        # Linear layers
        q = self.wq(query)  # (batch_size, seq_len, d_model)
        k = self.wk(key)    # (batch_size, seq_len, d_model)
        v = self.wv(value)  # (batch_size, seq_len, d_model)
        
        # Split heads
        q = self.split_heads(q, batch_size, sequence_length)  # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size, sequence_length)  # (batch_size, num_heads, seq_len, depth)
        v = self.split_heads(v, batch_size, sequence_length)  # (batch_size, num_heads, seq_len, depth)
        
        # Scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)  # (batch_size, num_heads, seq_len, depth)
        
        # Concatenate heads
        concat_attention = self.concat_heads(scaled_attention, batch_size, sequence_length)  # (batch_size, seq_len, d_model)
        
        # Final linear layer
        output = self.dense(concat_attention)  # (batch_size, seq_len, d_model)
        
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(embedding_dim)
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def __call__(self, embeddings, training=False, mask=None):
        # embeddings shape: (batch_size, seq_length, embedding_dim)
        
        # Multi-head attention
        attn_output = self.mha(embeddings, embeddings, embeddings, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(embeddings + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2  # shape: (batch_size, seq_length, embedding_dim)

class TransformerModel(tf.keras.Model):
    def __init__(self, embedding_dim, num_heads, dff, vocab_size, num_blocks=6, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        
        # Stack of transformer blocks
        self.transformer_blocks = [
            TransformerBlock(embedding_dim, num_heads, dff, dropout_rate) 
            for _ in range(num_blocks)
        ]
        
        # Final layer for next word prediction (outputs logits for each word in vocabulary)
        self.final_layer = Dense(vocab_size, activation='softmax')
    
    def __call__(self, embeddings, training=False, mask=None):
        # Pass through each transformer block in sequence
        x = embeddings
        for block in self.transformer_blocks:
            x = block(x, training=training, mask=mask)
        
        return self.final_layer(x)  


# Get preprocessed data with generator
PATH = "/Users/rayengallas/Desktop/Coding_projects/Project/shakespeare.txt"
BATCH_SIZE = 32
generator, vocab_size, num_sequences = Preprocessing_pipeline(file_path=PATH)(batch_size=BATCH_SIZE)

# Get shapes from first batch
X_batch, y_batch = next(generator)
embedding_dim = X_batch.shape[-1]

# Create the model
num_heads = 8
dff = 512  # dimension of feed-forward network

# Initialize the model
model = TransformerModel(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    dff=dff,
    vocab_size=vocab_size
)

class Training_pipeline:
    def __init__(self, model, generator, num_sequences, batch_size=32):
        self.model = model
        self.generator = generator
        self.batch_size = batch_size
        self.num_sequences = num_sequences
        self.steps_per_epoch = num_sequences // batch_size
        self.experiment_dir = "experiments"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
    def compile(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer, 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
    
    def save_experiment_results(self, history, test_results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "timestamp": timestamp,
            "config": {
                "batch_size": self.batch_size,
                "vocab_size": vocab_size,
                "embedding_dim": embedding_dim,
                "num_heads": num_heads
            },
            "training_history": history.history,
            "test_results": {
                "loss": float(test_results[0]),
                "accuracy": float(test_results[1])
            }
        }
        
        filename = f"{self.experiment_dir}/experiment_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
            
    def train(self, epochs=10):
        history = self.model.fit(
            self.generator,
            epochs=epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=None,
            validation_steps=None
        )
        
        test_results = self.model.evaluate(
            self.generator,
            steps=self.steps_per_epoch
        )
        
        print("\nTest Results:")
        print(f"Loss: {test_results[0]:.4f}")
        print(f"Accuracy: {test_results[1]:.4f}")
        
        self.save_experiment_results(history, test_results)
        
        return history
    
    def evaluate(self, test_generator, test_steps):
        return self.model.evaluate(
            test_generator,
            steps=test_steps
        )
    
    def __call__(self, epochs=10):
        self.compile()
        history = self.train(epochs)
        return history


def create_small_dataset(sequences, split_ratio=0.05, train_test_ratio=0.8):
    """
    Create a small dataset with only split_ratio% of the data
    Args:
        sequences: List of all sequences
        split_ratio: Percentage of total data to use (default 5%)
        train_test_ratio: Ratio of training to total data (default 80% train, 20% test)
    """
    # Calculate sizes
    total_size = len(sequences)
    small_size = int(total_size * split_ratio)
    train_size = int(small_size * train_test_ratio)
    
    # Randomly sample indices
    indices = np.random.permutation(total_size)
    small_indices = indices[:small_size]
    
    # Split into train and test
    train_indices = small_indices[:train_size]
    test_indices = small_indices[train_size:]
    
    # Create sequences
    train_sequences = [sequences[i] for i in train_indices]
    test_sequences = [sequences[i] for i in test_indices]
    
    return train_sequences, test_sequences


if __name__ == '__main__':
    print("Starting preprocessing...")
    preprocessor = Preprocessing_pipeline(file_path=PATH)
    
    # Get all sequences first
    raw_text = preprocessor.get_raw_text()
    tokenized_text = preprocessor.tokenize(raw_text)
    all_sequences = preprocessor.create_sequences(tokenized_text)
    
    # Create small dataset
    train_sequences, test_sequences = create_small_dataset(all_sequences)
    print(f"\nDataset sizes:")
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Testing sequences: {len(test_sequences)}")
    
    # Create generators
    train_generator = preprocessor.data_generator(train_sequences, BATCH_SIZE)
    test_generator = preprocessor.data_generator(test_sequences, BATCH_SIZE)
    
    # Initialize model
    model = TransformerModel(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dff=dff,
        vocab_size=vocab_size
    )
    
    # Create and run training pipeline
    trainer = Training_pipeline(
        model=model,
        generator=train_generator,
        num_sequences=len(train_sequences),
        batch_size=BATCH_SIZE
    )
    
    # Train the model
    history = trainer()
    
    # Evaluate
    test_steps = len(test_sequences) // BATCH_SIZE
    loss, accuracy = trainer.evaluate(test_generator, test_steps)
    print(f"\nTest Results:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")