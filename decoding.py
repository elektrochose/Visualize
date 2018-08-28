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


<<<<<<< HEAD
def index_to_labels(index):
	if type(index) == pd.core.indexes.base.Index:
		return [w for w in grid_scores.index.values]
	elif type(index) == pd.core.indexes.multi.MultiIndex:
		return [' '.join(w) for w in grid.index.values]
=======
>>>>>>> 3408c44961551156da1d6f8277af16a823d27347




<<<<<<< HEAD
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
=======

def plot_session_model_errors(session,
							  modelPredictions,
							  window_size = 5,
							  title = 'Decoding Errors per Session'):

	noTrials = len(session)
	learningCurve = np.zeros(noTrials)
	modelCurve = np.zeros(noTrials)

	for trial in range(noTrials):
		b_index = int(max(0, trial - np.floor(window_size/2)))
		t_index = int(min(noTrials, trial + np.ceil(window_size/2)))
		learningCurve[trial] = \
			np.float(np.sum(session['choice', 0].iloc[b_index:t_index]))\
			 										  / window_size
		modelCurve[trial] = \
			np.sum(modelPredictions[b_index:t_index]) / window_size

	#smoothing curves
	learningCurve = sig.savgol_filter(learningCurve, window_size, 2)
	modelCurve = sig.savgol_filter(modelCurve, window_size, 2)
	#determining where reversals happened
	blocks = session.groupby(axis=0, level='block')
	revPoints = np.cumsum([len(b) for l,b in blocks])
	#figuring out where model went wrong
	errorDecoding = np.nonzero(modelPredictions != session['choice', 0])[0]


	plt.figure()
	plt.plot(learningCurve, color = 'blue', label = 'session')
	plt.plot(modelCurve, color = 'purple', label = 'model')
	plt.xlim(0, noTrials)
	plt.ylim(0, 1)
	plt.yticks(np.linspace(0, 1, 11),
		  ['%1.1f' %w for w in list(np.linspace(0,1,11))])
	plt.xticks(np.arange(0, noTrials, 10),
		  list(np.arange(0, noTrials, 10)))
	plt.plot([0, noTrials],
			 [0.5, 0.5],
			 'k:', label = 'Chance')
	#plotting all the reversal lines
	for k in revPoints:
		plt.plot([k, k], [0, 1], 'green')
		plt.text(k, 1, 'Rev',
				  horizontalalignment = 'center',
				  color = 'green')

	#plot where the decoder made a mistake
	for l in errorDecoding:
		if l >= 0 and l < noTrials:
			plt.text(l, 0.9, 'x', color = 'red',
					 verticalalignment = 'center',
					 horizontalalignment = 'center')

	plt.xlabel('Trial #')
	plt.ylabel('P(Choosing W)')
	plt.legend()
	plt.title(title, FontSize=14)
	plt.show()
>>>>>>> 3408c44961551156da1d6f8277af16a823d27347
