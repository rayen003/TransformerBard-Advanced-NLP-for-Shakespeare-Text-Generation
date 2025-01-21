import tensorflow as tf
import numpy as np
from preprocessing_pipeline import Preprocessing_pipeline
from model import TransformerModel
import json
import os

class TextGenerator:
    def __init__(self, model, preprocessor, max_length=100):
        self.model = model
        self.preprocessor = preprocessor
        self.max_length = max_length
    
    def generate_text(self, prompt, temperature=1.0, num_words=50):
        """
        Generate text given a prompt.
        Args:
            prompt: Initial text to start generation
            temperature: Controls randomness (higher = more random)
            num_words: Number of words to generate
        """
        # Tokenize the prompt
        input_sequence = self.preprocessor.tokenize(prompt)
        
        # Generate text word by word
        generated_text = prompt
        for _ in range(num_words):
            # Prepare input sequence
            padded_sequence = self.preprocessor.pad_sequences([input_sequence])
            
            # Get model predictions
            predictions = self.model(padded_sequence)[0]
            predictions = predictions[-1] / temperature
            
            # Sample from the predictions
            predictions = tf.nn.softmax(predictions).numpy()
            predicted_id = np.random.choice(len(predictions), p=predictions)
            
            # Convert id back to word
            predicted_word = self.preprocessor.tokenizer.index_word.get(predicted_id, '')
            
            # Add the predicted word to the sequence
            input_sequence.append(predicted_id)
            generated_text += ' ' + predicted_word
            
            # Trim sequence if it gets too long
            if len(input_sequence) > self.max_length:
                input_sequence = input_sequence[-self.max_length:]
        
        return generated_text

    def format_output(self, generated_text):
        """Format the generated text for better readability"""
        # Add proper capitalization and basic punctuation
        sentences = generated_text.split('.')
        formatted_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Capitalize first letter
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence
                formatted_sentences.append(sentence + '.')
        
        return ' '.join(formatted_sentences)

def load_model_from_experiment(experiment_path):
    """
    Load model configuration from an experiment file.
    """
    with open(experiment_path, 'r') as f:
        experiment_data = json.load(f)
    
    config = experiment_data['config']
    
    # Create model with the same configuration
    model = TransformerModel(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        ff_dim=32,  # Default value
        num_transformer_blocks=4,  # Default value
        dropout=0.1  # Default value
    )
    
    # Load weights
    experiment_name = os.path.basename(experiment_path).replace('experiment_', 'weights_')
    weights_path = os.path.join('experiments/weights', experiment_name.replace('.json', '.weights.h5'))
    print(f"Loading weights from: {weights_path}")
    model.load_weights(weights_path)
    
    return model

if __name__ == '__main__':
    # Find the latest experiment file
    experiment_dir = "experiments"
    experiment_files = [f for f in os.listdir(experiment_dir) if f.endswith('.json')]
    latest_experiment = max(experiment_files, key=lambda x: os.path.getctime(os.path.join(experiment_dir, x)))
    experiment_path = os.path.join(experiment_dir, latest_experiment)
    
    print(f"Loading model from: {experiment_path}")
    
    # Load the model and create text generator
    model = load_model_from_experiment(experiment_path)
    
    # Create preprocessor and text generator
    preprocessor = Preprocessing_pipeline()
    text_generator = TextGenerator(model, preprocessor)
    
    # Generate text with different temperatures
    prompts = [
        "The king",
        "In the forest",
        "She looked",
    ]
    
    temperatures = [0.5, 1.0, 1.5]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        for temp in temperatures:
            print(f"\nTemperature: {temp}")
            generated_text = text_generator.generate_text(prompt, temperature=temp, num_words=50)
            print(generated_text)