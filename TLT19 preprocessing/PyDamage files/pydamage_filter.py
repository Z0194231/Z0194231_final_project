import pandas as pd 

# Load the results from PyDamage
dataset = pd.read_csv("pydamage_results/pydamage_results.csv")

# Apply the accuracy threshold of 0.67 as per the academic paper of PyDamage
dataset = dataset[dataset["predicted_accuracy"] >= 0.67]
dataset = dataset[dataset["qvalue"] <= 0.05]

# Save results to csv file
dataset.to_csv("pydamage_results/filtered_custom.csv", index=False)
