import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def index_to_labels(index):
	if type(index) == pd.core.indexes.base.Index:
		return [w for w in grid_scores.index.values]
	elif type(index) == pd.core.indexes.multi.MultiIndex:
		return [' '.join(w) for w in grid.index.values]




def plot_model_grid(df, vrange = [0.5, 0.8], title = 'Model Decoding'):

	row_labels = index_to_labels(df.index)
	col_labels = index_to_labels(df.columns)

	fig = plt.imshow(df.values, vmin = vrange[0], vmax = vrange[1])

	plt.xticks(np.linspace(0, len(row_labels) - 1, len(row_labels)),
											row_labels, rotation=90)
	plt.yticks(np.linspace(0, len(col_labels) - 1, len(col_labels)),
											col_labels, rotation=0)
	plt.ylabel('Model Trained on')
	plt.xlabel('Dataset Decoded')
	fig.axes.xaxis.tick_top()

	cbar = plt.colorbar()
	cbar.ax.get_yaxis().labelpad = 15
	cbar.ax.set_ylabel('Percent Trials Decoded', rotation = 270)
	plt.title(title + '\n\n', fontSize=18, y=1.3)
	plt.show()
	return
