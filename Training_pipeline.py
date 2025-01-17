import tensorflow as tf
from datetime import datetime
import os
import json
from model import TransformerModel
from preprocessing_pipeline import Preprocessing_pipeline

class Training_pipeline:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.experiment_dir = os.path.join('experiments', datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.model = None
        self.history = None
        
        # Model hyperparameters
        self.d_model = 256
        self.num_heads = 8
        self.dff = 512
        self.num_layers = 4
        self.dropout_rate = 0.1
    
    def create_model(self):
        """Create and compile the transformer model"""
        # Create model
        self.model = TransformerModel(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            input_dim=128,  # TinyBERT's embedding dimension
            vocab_size=self.preprocessor.vocabulary_size,
            dropout_rate=self.dropout_rate
        )
        
        # Learning rate schedule
        initial_learning_rate = 5e-5
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9
        )
        
        # Optimizer with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel architecture:")
        print("=" * 20)
        print(f"Number of layers: {self.num_layers}")
        print(f"Hidden size: {self.d_model}")
        print(f"Number of heads: {self.num_heads}")
        print(f"Feed-forward size: {self.dff}")
        print(f"Vocabulary size: {self.preprocessor.vocabulary_size}")
        print("=" * 20)
        
        return self.model
        
    def train(self, dataset, epochs=10, validation_split=0.2):
        """Train the model"""
        print(f"\nTraining model for {epochs} epochs with {validation_split*100}% validation split...")
        
        # Get dataset size
        dataset_size = sum(1 for _ in dataset)
        val_size = int(dataset_size * validation_split)
        train_size = dataset_size - val_size
        
        print(f"Dataset split:")
        print(f"- Total sequences: {dataset_size}")
        print(f"- Training sequences: {train_size}")
        print(f"- Validation sequences: {val_size}")
        
        # Split dataset
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size).take(val_size)
        
        # Add repeat to the datasets
        train_dataset = train_dataset.repeat()
        if val_size > 0:
            val_dataset = val_dataset.repeat()
        
        # Calculate steps per epoch - ensure at least 1 step
        steps_per_epoch = max(1, train_size // self.preprocessor.batch_size)
        validation_steps = max(1, val_size // self.preprocessor.batch_size) if val_size > 0 else None
        
        # Train model
        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset if val_size > 0 else None,
            validation_steps=validation_steps,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='loss',
                    patience=3,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.experiment_dir, 'best_model.keras'),
                    monitor='loss',
                    save_best_only=True
                )
            ]
        )

if __name__ == '__main__':
    PATH = "/Users/rayengallas/Desktop/Coding_projects/Project/data/shakespeare.txt"
    USE_PERCENTAGE = 0.01
    
    print(f"Initializing pipeline with {USE_PERCENTAGE*100}% of data...")
    
    # Initialize preprocessor with desired percentage
    preprocessor = Preprocessing_pipeline(
        file_path=PATH,
        max_length=50,
        batch_size=32,
        model_name='prajjwal1/bert-tiny',
        use_percentage=USE_PERCENTAGE
    )
    
    print(f"Preprocessor initialized with {preprocessor.use_percentage*100}% of data")
    print("Starting data preparation...")
    
    # Get preprocessed data
    dataset = preprocessor.prepare_data()
    
    print("\nInitializing training pipeline...")
    # Initialize trainer and create model
    trainer = Training_pipeline(preprocessor)
    model = trainer.create_model()
    
    print("\nStarting model training...")
    # Train model
    trainer.train(
        dataset=dataset,
        epochs=10,
        validation_split=0.2
    )