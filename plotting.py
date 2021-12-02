import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_tracks(df, tracks):
    for i in range(tracks):
        plt.plot(df.loc[df['particle_id']==i, 'x'], df.loc[df['particle_id']==i,'y'])
    plt.show()
