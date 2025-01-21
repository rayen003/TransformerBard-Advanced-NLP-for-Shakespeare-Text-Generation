import tensorflow as tf
from datetime import datetime
import os
import json
from model import TransformerModel
from preprocessing_pipeline import Preprocessing_pipeline

class Training_pipeline:
    def __init__(self, model, preprocessor, batch_size=32, learning_rate=0.001):
        self.model = model
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
    
    def prepare_datasets(self, full_dataset, train_size=0.8, val_size=0.2, use_percentage=0.1):
        """Split dataset into training and validation sets"""
        # Calculate total dataset size
        dataset_size = sum(1 for _ in full_dataset)
        print(f"\nTotal sequences available: {dataset_size}")
        
        # Take percentage of data
        subset_size = int(dataset_size * use_percentage)
        dataset_subset = full_dataset.take(subset_size)
        print(f"Using {use_percentage*100}% of data: {subset_size} sequences")
        
        # Calculate split sizes
        train_size = int(subset_size * train_size)
        val_size = int(subset_size * val_size)
        print(f"Training sequences: {train_size}")
        print(f"Validation sequences: {val_size}")
        
        # Split dataset
        train_dataset = dataset_subset.take(train_size)
        val_dataset = dataset_subset.skip(train_size).take(val_size)
        
        return train_dataset, val_dataset
    
    def train(self, dataset, num_epochs=10):
        """Train the model using Keras fit method"""
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = os.path.join('experiments', timestamp)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Split dataset
        train_dataset, val_dataset = self.prepare_datasets(dataset)
        
        # Create callbacks
        callbacks = [
            # Save model weights after each epoch
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(experiment_dir, 'weights.weights.h5'),
                save_weights_only=True,
                save_best_only=True,
                monitor='val_loss'
            ),
            # Save training history
            tf.keras.callbacks.CSVLogger(
                os.path.join(experiment_dir, 'training_history.csv')
            ),
            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(experiment_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=num_epochs,
            callbacks=callbacks
        )
        
        # Save training history as JSON
        with open(os.path.join(experiment_dir, 'experiment.json'), 'w') as f:
            json.dump(history.history, f, indent=4)
        
        print("\nTraining completed!")
        print(f"Experiment data saved to: {experiment_dir}")
        
        return history


if __name__ == '__main__':
    print("Initializing preprocessing pipeline...")
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
    
    # Initialize training pipeline
    trainer = Training_pipeline(
        model=model,
        preprocessor=preprocessor,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Train model
    history = trainer.train(dataset, num_epochs=10)