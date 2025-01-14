import tensorflow as tf
from transformers import AutoModel, TFAutoModel
import numpy as np
import os

class TinyBERTModel(tf.keras.Model):
    def __init__(self, model_name, max_length, vocab_size):
        super(TinyBERTModel, self).__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # Load pre-trained TinyBERT (convert from PyTorch weights)
        self.bert = TFAutoModel.from_pretrained(model_name, from_pt=True)
        
        # Add prediction head
        self.dense = tf.keras.layers.Dense(vocab_size, activation=None)
    
    def call(self, inputs, training=False):
        # Get BERT outputs
        bert_outputs = self.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            training=training
        )
        
        # Get the hidden states
        hidden_states = bert_outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
        
        # Use the last token's representation for prediction
        last_hidden_state = hidden_states[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Make prediction
        logits = self.dense(last_hidden_state)  # Shape: (batch_size, vocab_size)
        
        return logits
    
    def loss(self, targets, predictions):
        """Calculate loss between targets and predictions"""
        return tf.keras.losses.categorical_crossentropy(
            targets, predictions, from_logits=True
        )


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


class CustomLanguageModel(tf.keras.Model):
    def __init__(self, model_name, max_length, vocab_size, embed_dim=256, num_heads=4, ff_dim=512):
        super(CustomLanguageModel, self).__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # Load pre-trained BERT for embeddings only
        self.bert = TFAutoModel.from_pretrained(model_name, from_pt=True)
        
        # Freeze BERT weights
        self.bert.trainable = False
        
        # Add custom transformer blocks
        self.transformer_block1 = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim)
        
        # Project BERT hidden size to our embed_dim
        self.projection = tf.keras.layers.Dense(embed_dim)
        
        # Final prediction layer
        self.final_layer = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs, training=False):
        # Get BERT embeddings
        bert_outputs = self.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            training=False  # Always False since we're not training BERT
        )
        
        # Get hidden states and project to our dimension
        hidden_states = bert_outputs.last_hidden_state
        x = self.projection(hidden_states)
        
        # Pass through transformer blocks
        x = self.transformer_block1(x, training=training)
        x = self.transformer_block2(x, training=training)
        
        # Use the last token's representation for prediction
        last_hidden_state = x[:, -1, :]
        
        # Make prediction
        logits = self.final_layer(last_hidden_state)
        
        return logits
