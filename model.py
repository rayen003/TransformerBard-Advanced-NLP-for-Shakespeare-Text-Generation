import tensorflow as tf
from preprocessing_pipeline import Preprocessing_pipeline



class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_dim, vocab_size, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        
        # Token embedding
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        
        # Position embedding
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=50, output_dim=d_model)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        # Final prediction layers
        self.final_dense = tf.keras.layers.Dense(d_model, activation='relu')
        self.final_layer = tf.keras.layers.Dense(vocab_size)
        
        # Loss metric
        self.loss_metric = tf.keras.metrics.Mean(name='train_loss')
    
    def call(self, inputs, training=False):
        # Get sequence length and batch size
        seq_length = tf.shape(inputs)[1]
        batch_size = tf.shape(inputs)[0]
        
        # Create position indices
        positions = tf.range(start=0, limit=seq_length, delta=1)
        positions = tf.expand_dims(positions, axis=0)  # Add batch dimension
        positions = tf.tile(positions, [batch_size, 1])  # Repeat for each item in batch
        
        # Get token embeddings
        x = self.token_embedding(inputs)  # Shape: [batch_size, seq_length, d_model]
        
        # Add positional embeddings
        pos_emb = self.pos_embedding(positions)  # Shape: [batch_size, seq_length, d_model]
        x = x + pos_emb  # Shape: [batch_size, seq_length, d_model]
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)  # Shape: [batch_size, seq_length, d_model]
        
        # Process each position in the sequence
        x = self.final_dense(x)  # Shape: [batch_size, seq_length, d_model]
        
        # Make predictions for each position
        logits = self.final_layer(x)  # Shape: [batch_size, seq_length, vocab_size]
        
        # We only want the prediction for the last token in each sequence
        return logits[:, -1, :]  # Shape: [batch_size, vocab_size]

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate):
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            value_dim=d_model // num_heads
        )
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        # Layer normalization
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        # Multi-head attention
        attn_output = self.mha(query=x, key=x, value=x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

if __name__ == '__main__':
    print("Testing preprocessing pipeline...")
    PATH = "/Users/rayengallas/Desktop/Coding_projects/Project/data/shakespeare.txt"

    # Initialize preprocessor
    preprocessor = Preprocessing_pipeline(
        file_path=PATH,
        max_length=50,
        batch_size=32,
        model_name='prajjwal1/bert-tiny'
    )

    # Create dataset
    dataset = preprocessor.prepare_data()
    
    print("\nPreprocessing Results:")
    print("=" * 20)
    print(f"Vocabulary Size: {preprocessor.vocabulary_size}")
    print(f"Batch Size: {preprocessor.batch_size}")

    # Initialize model
    model = TransformerModel(
        num_layers=4,
        d_model=256,
        num_heads=8,
        dff=512,
        input_dim=50,  # max sequence length
        vocab_size=preprocessor.vocabulary_size,
        dropout_rate=0.1
    )

    # Test shapes through the model
    print("\nTesting model shapes:")
    print("=" * 20)
    
    for inputs, targets in dataset.take(1):
        print("\n1. Input Shapes:")
        print(f"- Raw input shape: {inputs['input_ids'].shape}")  # Should be (batch_size, seq_len)
        print(f"- Target shape: {targets.shape}")  # Should be (batch_size,)
        
        # Get model predictions
        predictions = model(inputs['input_ids'], training=False)
        
        print("\n2. Model Internal Shapes:")
        print(f"- After token embedding: {model.token_embedding(inputs['input_ids']).shape}")  # (batch_size, seq_len, d_model)
        
        # Create position indices
        positions = tf.range(start=0, limit=tf.shape(inputs['input_ids'])[1])
        positions = tf.expand_dims(positions, 0)
        print(f"- Position indices shape: {positions.shape}")  # (1, seq_len)
        
        # Get position embeddings
        pos_embeddings = model.pos_embedding(positions)
        print(f"- Position embeddings shape: {pos_embeddings.shape}")  # (1, seq_len, d_model)
        
        print("\n3. Final Output Shape:")
        print(f"- Model output shape: {predictions.shape}")  # (batch_size, vocab_size)
        print(f"- Expected target shape: {targets.shape}")  # (batch_size,)
        
        # Print value ranges
        print("\nValue ranges:")
        print(f"- Input values - min: {tf.reduce_min(inputs['input_ids'])}, max: {tf.reduce_max(inputs['input_ids'])}")
        print(f"- Output values - min: {tf.reduce_min(predictions):.4f}, max: {tf.reduce_max(predictions):.4f}")
        print(f"- Target values - min: {tf.reduce_min(targets)}, max: {tf.reduce_max(targets)}")
    print("=" * 20)