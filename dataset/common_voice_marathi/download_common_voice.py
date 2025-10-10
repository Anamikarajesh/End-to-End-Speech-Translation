from datasets import load_dataset
dataset = load_dataset("mozilla-foundation/common_voice_17_0", "mr")
dataset.save_to_disk("dataset/common_voice_marathi/local_copy")
print("Marathi Common Voice dataset downloaded successfully!")
