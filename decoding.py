import numpy as np
import pandas as pd
from scipy import signal as sig
import matplotlib.pyplot as plt
from behavioral_performance.utils import fileNames, fileNameLabels

def create_model_grid():
	#standard score dataframe
	iterables_row = [['DSR','PSR'],
					 ['FirstTraining', 'MidTraining', 'Saline',
					  'MPFC', 'OFC', 'Ipsi', 'Contra']]
	row_index  = pd.MultiIndex.from_product(iterables_row,
											names=['task','regime'])
	df = pd.DataFrame(np.full([len(row_index),len(row_index)], np.NaN),
													  index = row_index,
													  columns = row_index)
	return df


def plot_model_grid(df, vrange, title = 'Model Decoding'):

	labels = [df.index.levels[0][a] + df.index.levels[1][b] \
			 for a,b in zip(df.index.labels[0], df.index.labels[1])]

	noDataSets = len(labels)

	fig = plt.imshow(df.values, vmin = vrange[0], vmax = vrange[1])

	plt.xticks(np.linspace(0,noDataSets - 1, noDataSets),
											labels, rotation=90)
	plt.yticks(np.linspace(0,noDataSets - 1, noDataSets),
											labels, rotation=0)
	plt.ylabel('Model Trained on')
	plt.xlabel('Dataset Decoded')
	fig.axes.xaxis.tick_top()

	cbar = plt.colorbar()
	cbar.ax.get_yaxis().labelpad = 15
	cbar.ax.set_ylabel('Percent Trials Decoded', rotation = 270)
	plt.title(title + '\n\n', fontSize=18, y=1.3)
	plt.show()
	return


def flat_model_grid():
	return pd.DataFrame(np.zeros([len(fileNameLabels),len(fileNameLabels)]),
												  index = fileNameLabels,
												  columns = fileNameLabels)
def plot_flat_model_grid(df, vrange, title = 'Model Decoding'):
	labels = df.index.values
	noDataSets = len(labels)

	fig = plt.imshow(df.values, vmin = vrange[0], vmax = vrange[1])

	plt.xticks(np.linspace(0,noDataSets - 1, noDataSets),
											labels, rotation=90)
	plt.yticks(np.linspace(0,noDataSets - 1, noDataSets),
											labels, rotation=0)
	plt.ylabel('Model Trained on')
	plt.xlabel('Dataset Decoded')
	fig.axes.xaxis.tick_top()

	cbar = plt.colorbar()
	cbar.ax.get_yaxis().labelpad = 15
	cbar.ax.set_ylabel('Percent Trials Decoded', rotation = 270)
	plt.title(title + '\n\n', fontSize=18, y=1.3)
	plt.show()
	return


def index_to_labels(index):
	if type(index) == pd.core.indexes.base.Index:
		return [w for w in grid_scores.index.values]
	elif type(index) == pd.core.indexes.multi.MultiIndex:
		return [' '.join(w) for w in grid.index.values]




def plot_model_grid_general(df, vrange = [0.5, 0.8], title = 'Model Decoding'):

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
