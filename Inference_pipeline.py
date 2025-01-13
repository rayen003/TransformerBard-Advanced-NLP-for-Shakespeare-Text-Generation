import tensorflow as tf
from preprocessing_pipeline import Preprocessing_pipeline
from inference import TextGenerator, load_model_from_experiment
import os
import json
from datetime import datetime

class InferencePipeline:
    def __init__(self, experiment_path, weights_path=None):
        """
        Initialize the inference pipeline
        Args:
            experiment_path: Path to the experiment JSON file
            weights_path: Optional path to model weights
        """
        self.model = load_model_from_experiment(experiment_path)
        if weights_path:
            self.model.load_weights(weights_path)
        
        # Load preprocessor with same configuration as experiment
        with open(experiment_path, 'r') as f:
            config = json.load(f)
        
        self.preprocessor = Preprocessing_pipeline(
            file_path="/Users/rayengallas/Desktop/Coding_projects/Project/shakespeare.txt"
        )
        
        self.generator = TextGenerator(
            model=self.model,
            preprocessor=self.preprocessor
        )
        
        # Create directory for saving generations
        self.generations_dir = "generations"
        os.makedirs(self.generations_dir, exist_ok=True)
    
    def generate(self, prompt, temperature=1.0, num_words=50):
        """Generate text from a prompt"""
        generated_text = self.generator.generate_text(
            prompt=prompt,
            temperature=temperature,
            num_words=num_words
        )
        formatted_text = self.generator.format_output(generated_text)
        return formatted_text
    
    def save_generation(self, prompt, generated_text, temperature):
        """Save the generation with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.generations_dir, f"generation_{timestamp}.json")
        
        metadata = {
            "timestamp": timestamp,
            "prompt": prompt,
            "temperature": temperature,
            "generated_text": generated_text
        }
        
        with open(filename, "w") as f:
            json.dump(metadata, f, indent=4)
    
    def batch_generate(self, prompts, temperatures=[0.5, 0.7, 1.0], num_words=50):
        """Generate text for multiple prompts and temperatures"""
        results = []
        
        for prompt in prompts:
            prompt_results = []
            for temp in temperatures:
                generated = self.generate(prompt, temperature=temp, num_words=num_words)
                self.save_generation(prompt, generated, temp)
                
                prompt_results.append({
                    "temperature": temp,
                    "text": generated
                })
            
            results.append({
                "prompt": prompt,
                "generations": prompt_results
            })
        
        return results

if __name__ == "__main__":
    # Example usage
    experiment_path = "experiments/latest_experiment.json"  # Update with your path
    weights_path = None  # Update with your weights path
    
    pipeline = InferencePipeline(experiment_path, weights_path)
    
    # Test with some Shakespeare-style prompts
    test_prompts = [
        "To be or not to be",
        "All the world's a stage",
        "Friends, Romans, countrymen",
        "Romeo, Romeo, wherefore art thou",
        "Now is the winter of our discontent"
    ]
    
    results = pipeline.batch_generate(test_prompts)
    
    # Print results
    for result in results:
        print(f"\nPrompt: {result['prompt']}")
        for gen in result['generations']:
            print(f"\nTemperature {gen['temperature']}:")
            print(gen['text'])
            print("-" * 80)