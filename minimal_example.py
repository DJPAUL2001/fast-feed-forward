import cramming
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Note: force_download=True to prevent using the local cache
tokenizer = AutoTokenizer.from_pretrained("pbelcak/UltraFastBERT-1x11-long")
model = AutoModelForMaskedLM.from_pretrained("pbelcak/UltraFastBERT-1x11-long")

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

# Output keys are dict_keys(['loss', 'outputs'])
print(output["outputs"].shape)