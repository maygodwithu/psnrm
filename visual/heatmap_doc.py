import seaborn as sns
from matplotlib import pyplot as plt

flights = sns.load_dataset("flights")
flights = flights.pivot("month", "year", "passengers")
plt.figure(figsize=(10, 10))
ax = sns.heatmap(flights, annot=True, fmt="d")

plt.savefig('foo.png')

