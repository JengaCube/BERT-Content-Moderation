import torch
import transformers


tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

model = torch.load('Saved Model/model.safetensors')

text_list = ["Hello World!"]

model.eval()
for text in text_list:
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.sigmoid(logits)

    all_probabilities = []

    all_probabilities.append(probabilities)

print(all_probabilities)