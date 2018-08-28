import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib.patches import Ellipse, Circle
from matplotlib.collections import PatchCollection

idx = pd.IndexSlice
ROOT = os.environ['HOME'] + '/python/'


class AnimatedSession(object):
    def __init__(self, **kwargs):
        '''
        gonna be strict here. only accepting inputs as i specify them. we have
        title = 'title' - required
        session = session df - required
        model = model df - optional
        '''
        self.model_session_height = 0.6
        self.rat_session_height = 1.4
        for key in kwargs.keys():
            if key not in ['session','title','model']:
                raise ValueError('keys accepted: session, model, title.')

        self.title = kwargs.get('title')
        self.session = kwargs.get('session')
        self.model = kwargs.get('model')

        if not self.model is None:
            assert len(self.model) == len(self.session)

        # Setup the figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10,6))
        self.create_canvas()
        self.ani = animation.FuncAnimation(self.fig, self.animate,
                                           frames = len(self.session),
                                           interval = 500,
                                           blit = True,
                                           init_func = self.init)

    def get_trial_info(self, trial):
        if trial >= len(self.session):
            return
        else:
            cardinality = self.session['Choice'].iloc[trial]
            rewarded = self.session['AR'].iloc[trial]
            GA = self.session['GA'].iloc[trial]
            cardinality = (cardinality + 1) % 2
            GA = (GA + 1) % 2
            return cardinality, rewarded, GA

    def create_canvas(self):

        self.ax.set_xlim(( 0, 2))
        self.ax.set_ylim((0, 2))

        # calculate asymmetry of x and y axes:
        x0, y0 = self.ax.transAxes.transform((0, 0)) # lower left in pixels
        x1, y1 = self.ax.transAxes.transform((2, 2)) # upper right in pixes
        dx = x1 - x0
        dy = y1 - y0
        maxd = max(dx, dy)
        width = .25 * maxd / dx
        height = .25 * maxd / dy

        #background color and goal labels
        self.ax.plot([1,1],[0,2], color='k')
        self.ax.axvspan(0, 1, facecolor='purple', alpha=0.2)
        self.ax.axvspan(1, 2, facecolor='green', alpha=0.2)
        self.ax.text(0.35, 1.8, 'WEST', fontsize=24)
        self.ax.text(1.4, 1.8, 'EAST', fontsize=24)
        self.ax.tick_params(axis='both', bottom=False, left=False)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])


        #these parameters will change every trial
        cardinality = 1
        #dynamic elements
        #trial information at bottom-left corner
        self.trial_text = self.ax.text(1.6, 0.1, 'Trial 1', fontsize=20)
        #current goal star
        self.goal = self.ax.scatter(0.7 + cardinality,
                                    1.86, s=580, marker='*', color='y')
        #correct ellipse
        self.correct = self.ax.add_artist(
                              Ellipse((0.5 + cardinality,
                                       self.rat_session_height),
                              width,
                              height,
                              facecolor='blue',
                              edgecolor='k',
                              linewidth=3,
                              alpha=0.5))


        #error triangle
        self.error = \
            self.ax.add_patch(plt.Polygon([[0.4 + cardinality,
                                            self.rat_session_height + 0.2],
                                          [0.6 + cardinality,
                                           self.rat_session_height + 0.2],
                                          [0.5 + cardinality,
                                           self.rat_session_height - 0.2]],
                                          facecolor='red',
                                          edgecolor='k'))

        self.model_choice = self.ax.add_artist(
                              Ellipse((0.5 + cardinality,
                                       self.model_session_height),
                              width,
                              height,
                              facecolor='none',
                              edgecolor='k',
                              linewidth=3,
                              alpha=0.5))
        if self.model is None:
            self.model_choice.set_visible(False)

        plt.title(self.title, fontsize=30)
        return self.trial_text, self.goal, self.correct, self.error, self.model_choice


    def init(self):
        #will include some polygons i suppose
        self.trial_text.set_text('')
        return self.trial_text,

    def animate(self, i):
        self.trial_text.set_text('Trial %i' %i)
        cardinality, rewarded, GA = self.get_trial_info(i)
        if not self.model is None:
            model_choice = self.model['Choice'].iloc[i]
            model_choice = (model_choice + 1) % 2
        nudgeX = (np.random.random() - 0.5)* 0.1
        nudgeY = (np.random.random() - 0.5)* 0.1
        if rewarded == 0:
            self.error.set_visible(True)
            self.error.set_xy([[0.4 + cardinality + nudgeX,
                                self.rat_session_height + 0.2 + nudgeY],
                              [0.6 + cardinality + nudgeX,
                               self.rat_session_height + 0.2 + nudgeY],
                              [0.5 + cardinality + nudgeX,
                               self.rat_session_height - 0.2 + nudgeY]])
            self.correct.set_visible(False)
        else:
            self.correct.set_visible(True)
            self.correct.center = (0.5 + cardinality + nudgeX,
                                   self.rat_session_height + nudgeY)
            self.error.set_visible(False)

        self.goal.set_offsets([0.7 + GA, 1.86])

        if not self.model is None:
            self.model_choice.center = (0.5 + model_choice,
                                        self.model_session_height)


        return self.trial_text, self.correct, self.error, self.goal, self.model_choice

    def save_to_file(self):
        target = ROOT + 'Results/session_animation/' + self.title + '.mp4'
        self.ani.save(target, fps=2,
                    extra_args=['-vcodec', 'h264',
                  '-pix_fmt', 'yuv420p'])
    def show(self):
        plt.show()
        HTML(self.ani.to_html5_video())
if __name__ == '__main__':
    #load a dataset
    sessions = pickle.load(open(ROOT + \
            'DATA_structures/session_dataframes/PSR_SESSIONS_DATAFRAME.p','rb'))
    #get a single session
    session = sessions.loc[idx['57','PL-muscimol',:,:],:]
    t1 = time.time()
    a = AnimatedSession(session=session, title='PL-muscimol')
    t2 = time.time()
    a.save_to_file()
    t3 = time.time()
    print 'time to create object:%.2f' %(t2-t1)
    print 'time to save video: %.2f' %(t3 - t2)
