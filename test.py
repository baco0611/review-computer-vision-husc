import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = [
    [995, 0],
    [1000, 0]
]

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"], annot_kws={"size": 22})
plt.xlabel('Predicted labels', fontsize="14")
plt.ylabel('True labels', fontsize="14")
plt.title('Confusion Matrix', fontsize="14")
# plt.xticks(fontsize="14")
# plt.yticks(fontsize="14")
plt.savefig(f'VGG11_only_cat')
plt.show()