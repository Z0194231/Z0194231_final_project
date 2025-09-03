import pandas as pd
import matplotlib.pyplot as plt
import os


#Open the dataset created by PyDamage
dataset = pd.read_csv("pydamage_results/pydamage_results.csv")

#Filter dataset by coverage
dataset_filtered = dataset[dataset['nb_reads_aligned'] >= 10]

#Find contigs that are under the significance threshold of qvalue < 0.05
dataset_significant = dataset_filtered[dataset_filtered['qvalue'] < 0.05]

# Sort by damage in descending manner
dataset_sorted = dataset_significant.sort_values('damage_model_pmax', ascending=False)

#Create a summary table 
cols = ['reference', 'damage_model_pmax', 'damage_model_p', 'coverage',
                'nb_reads_aligned', 'qvalue']
path = os.path.join("pydamage_figure", "top_damage_contigs.tsv")

#Save summary table to tsv format
dataset_sorted[cols].to_csv(path, sep="\t", index=False)

#Plot the Damage VS Coverage graph for visual evaluation 
plt.figure(figsize=(8,6))
plt.scatter(dataset_sorted['coverage'], dataset_sorted['damage_model_pmax'], c='red', alpha=0.7)
plt.xlabel("Coverage")
plt.ylabel("Damage at 5' end (pmax)")
plt.title("aDNA Damage vs Coverage")
plt.tight_layout()
plt.savefig(os.path.join("pydamage_figure", "damage_vs_coverage.png"))
plt.close()
