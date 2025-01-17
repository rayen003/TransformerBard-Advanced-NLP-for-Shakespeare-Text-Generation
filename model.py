import tensorflow as tf

class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_dim, vocab_size, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        # Project input embeddings to model dimension
        self.input_projection = tf.keras.layers.Dense(d_model)
        
        # Position embedding
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=50, output_dim=d_model)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        # Final prediction layer
        self.final_layer = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs, training=False):
        # Get sequence length
        seq_length = tf.shape(inputs)[1]
        
        # Create position indices
        positions = tf.range(start=0, limit=seq_length, delta=1)
        
        # Project input embeddings to model dimension
        x = self.input_projection(inputs)
        
        # Add positional embeddings
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        # Get the last token's representation
        x = x[:, -1, :]
        
        # Make prediction
        return self.final_layer(x)

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