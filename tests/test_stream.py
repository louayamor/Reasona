from datasets import load_dataset

dataset_name = "PleIAs/SYNTH"
split = "train"

ds = load_dataset(dataset_name, split=split, streaming=True)

for i, sample in enumerate(ds):
    print(f"Sample {i+1}: {sample}")
    if i >= 4:  # read only 5 samples
        break
