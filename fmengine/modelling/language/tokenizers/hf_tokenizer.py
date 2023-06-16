from transformers import AutoTokenizer

def get_tokenizers(tokenizer_name):
    return AutoTokenizer.from_pretrained(tokenizer_name)