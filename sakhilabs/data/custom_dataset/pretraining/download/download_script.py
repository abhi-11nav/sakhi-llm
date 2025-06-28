import json

from datasets import load_dataset

# Load the dataset from Hugging Face cache
custom_path = "YOUR CUSTOM PATH"
dataset = load_dataset(
    "oscar-corpus/oscar", "unshuffled_deduplicated_te", cache_dir=custom_path
)
telugu_data = dataset["train"]

# Convert dataset to list of dictionaries
data_list = telugu_data[:]

# Save as a pure JSON array
output_path = "/Users/abhinav/Desktop/telugu_dataset/telugu_oscar_pure.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

print(f"Saved pure JSON to: {output_path}")
