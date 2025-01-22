import tensorflow as tf
from preprocessing_pipeline import Preprocessing_pipeline



class TransformerModel(tf.keras.Model):
    def __init__(self, vocab_size, max_length=32):
        super().__init__()
        
        # Model dimensions - minimal for fast training
        self.d_model = 32
        self.num_heads = 1
        self.dff = 64
        self.num_layers = 1
        self.dropout_rate = 0.1
        
        # Embeddings
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, self.d_model)
        self.pos_embedding = tf.keras.layers.Embedding(max_length, self.d_model)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(self.d_model, self.num_heads, self.dff, self.dropout_rate)
            for _ in range(self.num_layers)
        ]
        
        # Final layers
        self.final_layer = tf.keras.layers.Dense(vocab_size)
        
        # Metrics
        self.loss_metric = tf.keras.metrics.Mean(name='train_loss')
    
    def call(self, inputs, training=False):
        try:
            # Reshape inputs if necessary
            if len(inputs.shape) == 3:  # (batch, inner_batch, seq_len)
                batch_size = tf.shape(inputs)[0]
                inner_batch = tf.shape(inputs)[1]
                seq_length = tf.shape(inputs)[2]
                inputs = tf.reshape(inputs, [-1, seq_length])  # Flatten batches
            else:
                seq_length = tf.shape(inputs)[1]
                batch_size = tf.shape(inputs)[0]
            
            # Create position indices
            positions = tf.range(start=0, limit=seq_length, delta=1)
            positions = tf.expand_dims(positions, axis=0)
            positions = tf.tile(positions, [tf.shape(inputs)[0], 1])
            
            # Get embeddings
            x = self.token_embedding(inputs)  # [batch_size, seq_len, d_model]
            pos_emb = self.pos_embedding(positions)  # [batch_size, seq_len, d_model]
            
            # Add embeddings
            x = x + pos_emb  # [batch_size, seq_len, d_model]
            
            # Pass through transformer blocks
            for block in self.transformer_blocks:
                x = block(x, training=training)  # [batch_size, seq_len, d_model]
            
            # Get predictions for the last token
            x = self.final_layer(x)  # [batch_size, seq_len, vocab_size]
            x = x[:, -1, :]  # [batch_size, vocab_size]
            
            # No need to reshape back - we want [batch_size, vocab_size] for sparse categorical crossentropy
            return x
            
        except Exception as e:
            tf.print("Error in model.call():", e)
            tf.print("Input shape:", tf.shape(inputs))
            raise

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
        vocab_size=preprocessor.vocabulary_size,
        max_length=50
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