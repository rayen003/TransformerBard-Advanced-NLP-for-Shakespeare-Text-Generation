# TransformerBard: Advanced NLP for Shakespeare Text Generation

A sophisticated implementation of a transformer-based language model for generating Shakespeare-style text. This project showcases advanced natural language processing techniques, attention mechanisms, and deep learning architectures.

## Overview
This repository implements and experiments with various transformer architectures for creative text generation. The project demonstrates proficiency in:
- Advanced NLP techniques
- Transformer architecture implementation
- Attention mechanism optimization
- Large language model training
- Systematic experimentation and analysis

## Recent Updates (2025-01-17)

### Architecture Improvements
1. **Enhanced Model Architecture**
   - Added intermediate dense layers (512, 256 units)
   - Implemented dropout layers (0.1) for regularization
   - Added gradient clipping (clipnorm=1.0)

2. **Learning Rate Optimization**
   - Implemented exponential decay schedule
   - Initial learning rate: 5e-5
   - Decay steps: 1000
   - Decay rate: 0.9

3. **Training Pipeline**
   - Improved dataset handling with proper train/validation split
   - Enhanced experiment logging with detailed metrics
   - Added early stopping with patience=3

### Experiment Tracking
Now tracking detailed experiment information including:
- Data usage statistics
- Model configuration
- Training history
- Validation metrics
- Architecture details

## Project Structure
```
.
├── README.md
├── data/
│   └── shakespeare.txt
├── src/
│   ├── preprocessing_pipeline.py
│   ├── training_pipeline.py
│   ├── inference_pipeline.py
│   └── inference.py
├── experiments/
│   └── README.md
└── requirements.txt
```

## Experiments
Each experiment is documented in the `experiments/` directory with its own configuration, results, and analysis.

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python src/training_pipeline.py`

## Usage
1. **Preprocessing**:
   ```python
   preprocessor = Preprocessing_pipeline(
       file_path="data/shakespeare.txt",
       max_length=50,
       batch_size=32
   )
   dataset, size = preprocessor.prepare_data(use_percentage=0.2)
   ```

2. **Training**:
   ```python
   trainer = Training_pipeline(preprocessor)
   trainer.create_model()
   trainer.train(
       dataset=dataset,
       dataset_size=size,
       train_split=0.9,
       epochs=10
   )
   ```

## Next Steps
1. Experiment with larger context windows
2. Implement more sophisticated data augmentation
3. Try different model architectures
4. Add temperature-based sampling for inference

