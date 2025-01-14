from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


example = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(example)
print(tokens)