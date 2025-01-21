import tensorflow as tf
from datetime import datetime
import os
import json
from model import TransformerModel
from preprocessing_pipeline import Preprocessing_pipeline
import numpy as np

class Training_pipeline:
    def __init__(self, model, preprocessor, batch_size=32, learning_rate=0.001):
        self.model = model
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # Initialize metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        
        # Initialize experiment tracking
        self.experiment_dir = None
        self.experiment_metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    @tf.function
    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(inputs['input_ids'], training=True)
            
            # Calculate loss
            loss = self.loss_fn(targets, predictions)
        
        # Calculate gradients and update weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss(loss)
        self.train_accuracy(targets, predictions)
        
        return loss
    
    @tf.function
    def val_step(self, inputs, targets):
        # Forward pass
        predictions = self.model(inputs['input_ids'], training=False)
        
        # Calculate loss
        loss = self.loss_fn(targets, predictions)
        
        # Update metrics
        self.val_loss(loss)
        self.val_accuracy(targets, predictions)
        
        return loss
    
    def train(self, epochs=10, validation_split=0.2, experiment_dir=None):
        """Train the model for a specified number of epochs"""
        print("\nStarting model training...")
        print(f"\nTraining model for {epochs} epochs with {validation_split*100}% validation split...")
        
        # Get dataset
        dataset = self.preprocessor.prepare_data()
        
        # Calculate dataset sizes
        total_size = sum(1 for _ in dataset)
        train_size = int(total_size * (1 - validation_split))
        val_size = total_size - train_size
        
        print("Dataset split:")
        print(f"- Total sequences: {total_size}")
        print(f"- Training sequences: {train_size}")
        print(f"- Validation sequences: {val_size}")
        
        # Split dataset
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
        
        # Set up experiment directory
        if experiment_dir:
            self.experiment_dir = experiment_dir
            os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Reset metrics
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()
            
            # Training
            for batch in train_dataset:
                inputs, targets = batch
                loss = self.train_step(inputs, targets)
                
            # Validation
            for batch in val_dataset:
                inputs, targets = batch
                loss = self.val_step(inputs, targets)
            
            # Print metrics
            template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.4f}'
            print(template.format(epoch + 1,
                                self.train_loss.result(),
                                self.train_accuracy.result(),
                                self.val_loss.result(),
                                self.val_accuracy.result()))
            
            # Store metrics
            self.experiment_metrics['train_loss'].append(float(self.train_loss.result()))
            self.experiment_metrics['train_accuracy'].append(float(self.train_accuracy.result()))
            self.experiment_metrics['val_loss'].append(float(self.val_loss.result()))
            self.experiment_metrics['val_accuracy'].append(float(self.val_accuracy.result()))
            
            # Save experiment metrics
            if self.experiment_dir:
                metrics_file = os.path.join(self.experiment_dir, 'experiment.json')
                with open(metrics_file, 'w') as f:
                    json.dump(self.experiment_metrics, f, indent=4)

if __name__ == '__main__':
    PATH = "/Users/rayengallas/Desktop/Coding_projects/Project/data/shakespeare.txt"
    
    print("Initializing pipeline with 1.0% of data...")
    # Initialize preprocessor
    preprocessor = Preprocessing_pipeline(
        file_path=PATH,
        use_percentage=0.01,
        max_length=50,
        batch_size=32
    )
    
    print("\nInitializing training pipeline...")
    # Initialize trainer and create model
    model = TransformerModel(
        num_layers=4,
        d_model=256,
        num_heads=8,
        dff=512,
        input_dim=128,  # TinyBERT's embedding dimension
        vocab_size=preprocessor.vocabulary_size,
        dropout_rate=0.1
    )
    trainer = Training_pipeline(model, preprocessor)
    
    print("\nStarting model training...")
    # Train model
    trainer.train(
        epochs=10,
        validation_split=0.2,
        experiment_dir=os.path.join('experiments', datetime.now().strftime("%Y%m%d-%H%M%S"))
    )