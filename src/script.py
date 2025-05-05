import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('../data/dataset.csv')

df.plot()
plt.savefig('../results/plot.png', dpi=300)
