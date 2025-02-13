import tensorflow as tf
from datetime import datetime
import os
import json
from model import TransformerModel
from preprocessing_pipeline import Preprocessing_pipeline

# Set environment variables 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizer warnings

class Training_pipeline:
    def __init__(self, model, preprocessor, batch_size=64, learning_rate=0.001):
        self.model = model
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.initial_learning_rate = learning_rate
        
        # Enable mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=learning_rate,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1
        )
        
        # Compile model with mixed precision
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=0.01,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
            jit_compile=True  # Enable XLA compilation
        )
    
    def optimize_dataset(self, dataset):
        """Apply performance optimizations to dataset"""
        def reshape_targets(x, y):
            return {'input_ids': x}, tf.reshape(y, [-1])
            
        return dataset.cache() \
            .shuffle(buffer_size=10000) \
            .batch(self.batch_size) \
            .map(reshape_targets) \
            .prefetch(tf.data.AUTOTUNE)
    
    def prepare_datasets(self, full_dataset, train_size=0.8, val_size=0.2):
        """Split dataset into training and validation sets"""
        # Calculate total dataset size using cardinality
        dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
        print(f"\nTotal sequences in dataset: {dataset_size}")
        
        if dataset_size == 0:
            raise ValueError("Input dataset is empty! Check the preprocessing pipeline.")
        
        # Calculate split sizes (directly using the full dataset)
        train_size = int(dataset_size * train_size)
        val_size = int(dataset_size * val_size)
        print(f"Training sequences: {train_size}")
        print(f"Validation sequences: {val_size}")
        
        # Take data first, then split
        train_dataset = full_dataset.take(train_size)
        val_dataset = full_dataset.skip(train_size).take(val_size)
        
        def prepare_batch(inputs, targets):
            # Ensure targets are 1D
            return inputs, tf.reshape(targets, [-1])
        
        # Optimize dataset pipeline
        train_dataset = (train_dataset
            .batch(self.batch_size, drop_remainder=True)  # Batch first for speed
            .map(prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .shuffle(buffer_size=100)  # Smaller shuffle buffer
            .prefetch(tf.data.AUTOTUNE))
        
        val_dataset = (val_dataset
            .batch(self.batch_size, drop_remainder=True)
            .map(prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .prefetch(tf.data.AUTOTUNE))
        
        # Debug: Check if datasets are empty
        train_size = sum(1 for _ in train_dataset)
        val_size = sum(1 for _ in val_dataset)
        print("\nAfter batching:")
        print(f"Training batches: {train_size}")
        print(f"Validation batches: {val_size}")
        
        return train_dataset, val_dataset

    def train(self, dataset, num_epochs=5):
        """Train the model using Keras fit method"""
        try:
            # Create experiment directory
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_dir = os.path.join('experiments', timestamp)
            os.makedirs(experiment_dir, exist_ok=True)
            
            print(f"\nStarting training for {num_epochs} epochs")
            
            # Split dataset
            train_dataset, val_dataset = self.prepare_datasets(dataset)
            
            # Check dataset sizes
            train_size = sum(1 for _ in train_dataset)
            val_size = sum(1 for _ in val_dataset)
            print(f"\nDataset sizes:")
            print(f"Training batches: {train_size}")
            print(f"Validation batches: {val_size}")
            
            if train_size == 0 or val_size == 0:
                raise ValueError("Dataset is empty! Check the use_percentage parameter and dataset preparation.")
            
            # Simplified callbacks for speed
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(experiment_dir, 'weights.keras'),
                    save_best_only=True,
                    monitor='val_loss',
                    save_weights_only=True  # Faster than saving full model
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=2,  # Reduced patience
                    restore_best_weights=True
                )
            ]
            
            # Train model with specified epochs
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=num_epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            print("\nTraining completed!")
            print(f"Model weights saved to: {os.path.join(experiment_dir, 'weights.keras')}")
            
            # Save experiment configuration and history
            experiment_config = {
                "timestamp": timestamp,
                "num_epochs": num_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.initial_learning_rate,
                "model_params": {
                    "num_layers": self.model.num_layers,
                    "d_model": self.model.d_model,
                    "num_heads": self.model.num_heads,
                    "dff": self.model.dff,
                    "dropout_rate": self.model.dropout_rate
                }
            }
            
            # Save config and history
            config_path = os.path.join(experiment_dir, 'experiment_config.json')
            history_path = os.path.join(experiment_dir, 'training_history.json')
            
            with open(config_path, 'w') as f:
                json.dump(experiment_config, f, indent=4)
            
            with open(history_path, 'w') as f:
                history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
                json.dump(history_dict, f, indent=4)
            
            print(f"Configuration saved to: {config_path}")
            print(f"Training history saved to: {history_path}")
            
            return history
            
        except Exception as e:
            print("Error during training:", str(e))
            raise


if __name__ == '__main__':
    print("Testing preprocessing pipeline...")
    PATH = "/Users/rayengallas/Desktop/Coding_projects/Project/data/shakespeare.txt"
    
    # Initialize preprocessor
    preprocessor = Preprocessing_pipeline(
        file_path=PATH,
        max_length=32,
        batch_size=64,
        model_name='prajjwal1/bert-tiny',
        use_percentage=0.01 # Use 10% of data
    )
    
    # Create dataset
    dataset = preprocessor.prepare_data()
    
    # Initialize model
    model = TransformerModel(
        vocab_size=preprocessor.vocabulary_size,
        max_length=preprocessor.max_length
    )
    
    # Initialize training pipeline
    trainer = Training_pipeline(
        model=model,
        preprocessor=preprocessor,
        batch_size=64,
        learning_rate=0.001
    )
    
    # Train model
    history = trainer.train(dataset, num_epochs=10)