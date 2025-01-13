import tensorflow as tf
import numpy as np

class TransformerModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_heads=8, ff_dim=256, num_transformer_blocks=6, dropout=0.1, max_sequence_length=100):
        super(TransformerModel, self).__init__()
        
        self.pos_encoding = self._positional_encoding(max_sequence_length, embedding_dim)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(embedding_dim, activation='relu')
        ])
        
        self.transformer_blocks = []
        for _ in range(num_transformer_blocks):
            self.transformer_blocks.append(
                TransformerBlock(embedding_dim, num_heads, ff_dim, dropout)
            )
        
        self.final_layer = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(vocab_size)
        ])
    
    def _positional_encoding(self, max_len, d_model):
        pos_encoding = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs, training=False):
        # inputs shape: (batch_size, seq_len, embedding_dim)
        seq_len = tf.shape(inputs)[1]
        
        # Add positional encoding
        x = inputs * tf.math.sqrt(tf.cast(tf.shape(inputs)[-1], tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        # Apply dropout and encoding
        x = self.dropout(x, training=training)
        x = self.encoder(x)
        
        # Pass through transformer blocks with residual connections
        for transformer_block in self.transformer_blocks:
            x = x + transformer_block(x, training=training)  # Added residual connection
        
        # Final layer
        return self.final_layer(x)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,  # Scaled key dimension
            value_dim=embed_dim // num_heads  # Scaled value dimension
        )
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="gelu"),  # Changed to GELU
            tf.keras.layers.Dense(embed_dim)
        ])
        
        # Layer normalization and dropout
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, inputs, training=False):
        # Layer normalization and attention
        x = self.layernorm1(inputs)
        attn_output = self.att(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output  # Residual connection
        
        # Layer normalization and feed-forward
        x = self.layernorm2(out1)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output  # Residual connection
