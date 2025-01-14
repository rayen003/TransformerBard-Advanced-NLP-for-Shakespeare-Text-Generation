import tensorflow as tf
from datetime import datetime
import os
from model import CustomLanguageModel
from Preprocessing_pipeline import Preprocessing_pipeline

class Training_pipeline:
    def __init__(self, model_name='prajjwal1/bert-tiny', max_length=50, batch_size=32):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.experiment_dir = os.path.join('experiments', timestamp)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize preprocessor
        self.preprocessor = Preprocessing_pipeline(
            file_path="/Users/rayengallas/Desktop/Coding_projects/Project/data/shakespeare.txt",
            max_length=max_length,
            batch_size=batch_size,
            model_name=model_name
        )
    
    def create_model(self):
        """Create and compile model"""
        self.model = CustomLanguageModel(
            model_name=self.model_name,
            max_length=self.max_length,
            vocab_size=self.preprocessor.vocab_size,
            embed_dim=256,  # Smaller than BERT's hidden size
            num_heads=4,    # Fewer attention heads
            ff_dim=512      # Smaller feed-forward dimension
        )
        
        # Use a slightly higher learning rate since we're training from scratch
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, epochs=10):
        """Train the model"""
        # Get training data
        train_dataset, val_dataset = self.preprocessor.prepare_training_data(use_percentage=0.15)
        
        # Simple early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Train
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[early_stopping]
        )
        
        # Save best model
        self.model.save_weights(os.path.join(self.experiment_dir, 'model.weights.h5'))
        
        return history


if __name__ == '__main__':
    # Allow memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Initialize trainer
    trainer = Training_pipeline()
    
    # Create model
    print("\nCreating model...")
    trainer.create_model()
    
    # Train model
    print("\nStarting training...")
    trainer.train(epochs=10)
    
    print(f"\nTraining completed! Model saved in: {trainer.experiment_dir}")