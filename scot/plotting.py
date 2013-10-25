# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Object oriented API to SCoT """

import numpy as np
from .varica import mvarica
from .plainica import plainica
from .datatools import dot_special
from .connectivity import Connectivity
from . import var
from eegtopo.topoplot import Topoplot

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    _have_pyplot = True
except ImportError:
    _have_pyplot = False


def show_plots():
    plt.show()
    

def prepare_topoplots(topo, values):
    
    values = np.atleast_2d(values)
    
    topomaps = []
    
    for i in range(values.shape[0]):
        topo.set_values(values[i,:])
        topo.create_map()
        topomaps.append(topo.get_map())
        
    return topomaps
    

def plot_topo(axis, topo, topomap, crange=None):
    topo.set_map(topomap)
    h = topo.plot_map(axis, crange=crange)
    topo.plot_locations(axis)
    topo.plot_head(axis)
    return h

    
def plot_sources(topo, mixmaps, unmixmaps, global_scale=None, fig=None):
    """ global_scale:
           None - scales each topo individually
           1-99 - percentile of maximum of all plots
    """
    if not _have_pyplot:
        raise ImportError("matplotlib.pyplot is required for plotting")
    
    urange, mrange = None, None
    
    M = len(mixmaps)
    
    if global_scale:        
        tmp = np.asarray(unmixmaps)
        tmp = tmp[np.logical_not(np.isnan(tmp))]     
        umax = np.percentile(np.abs(tmp), global_scale)
        umin = -umax
        urange = [umin,umax]
        
        tmp = np.asarray(mixmaps)
        tmp = tmp[np.logical_not(np.isnan(tmp))]   
        mmax = np.percentile(np.abs(tmp), global_scale)
        mmin = -mmax
        mrange = [umin,umax]
        
    Y = np.floor(np.sqrt(M*3/4))
    X = np.ceil(M/Y)
    
    if fig is None:
        fig = plt.figure()
    
    axes = []
    for m in range(M):
        axes.append(fig.add_subplot(2*Y, X, m+1))
        h1 = plot_topo(axes[-1], topo, unmixmaps[m], crange=urange)
        axes[-1].set_title(str(m))
        
        axes.append(fig.add_subplot(2*Y, X, M+m+1))
        h2 = plot_topo(axes[-1], topo, mixmaps[m], crange=urange)
        axes[-1].set_title(str(m))
        
    for a in axes:
        a.set_yticks([])
        a.set_xticks([])
        a.set_frame_on(False)
        
    axes[0].set_ylabel('Unmixing weights')
    axes[1].set_ylabel('Scalp projections')
    
    #plt.colorbar(h1, plt.subplot(2, M+1, M+1))
    #plt.colorbar(h2, plt.subplot(2, M+1, 0))
    
    return fig
    
    
def plot_connectivity_spectrum(A, fs=2, freq_range=(-np.inf, np.inf), topo=None, topomaps=None, fig=None):
    A = np.atleast_3d(A)
    [N,M,F] = A.shape
    freq = np.linspace(0, fs/2, F)

    lowest, highest = np.inf, -np.inf
    left = max(freq_range[0], freq[0])
    right = min(freq_range[1], freq[-1])
    
    if fig is None:
        fig = plt.figure()
    
    axes = []
    for n in range(N):
        arow = []
        for m in range(M):
            ax = fig.add_subplot(N, M, m+n*M+1)
            arow.append(ax)
            
            if n == m:
                if topo:
                    plot_topo(ax, topo, topomaps[m])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_frame_on(False)
            else:                
                ax.plot(freq, A[n,m,:])
                lowest = min(lowest, np.min(A[n,m,:]))
                highest = max(highest, np.max(A[n,m,:]))
                ax.set_xlim(0, fs/2)
        axes.append(arow)
        
    for n in range(N):
        for m in range(M):
            if n == m:
                pass
            else:
                axes[n][m].xaxis.set_major_locator(MaxNLocator(max(1,7-N)))
                axes[n][m].yaxis.set_major_locator(MaxNLocator(max(1,7-M)))
                axes[n][m].set_ylim(lowest, highest)
                axes[n][m].set_xlim(left, right)
                if 0 < n < N-1:
                    axes[n][m].set_xticks([])
                if 0 < m < M-1:
                    axes[n][m].set_yticks([])                    
        axes[n][0].yaxis.tick_left()
        axes[n][-1].yaxis.tick_right()
        
    for m in range(M):
        axes[0][m].xaxis.tick_top()
        axes[-1][m].xaxis.tick_bottom()
        
    fig.text(0.5, 0.025, 'frequency', horizontalalignment='center')
    fig.text(0.05, 0.5, 'magnitude', horizontalalignment='center', rotation='vertical')
    
    return fig
    
def plot_connectivity_timespectrum(A, fs=2, crange=None, freq_range=(-np.inf, np.inf), time_range=None, topo=None, topomaps=None, fig=None):
    A = np.asarray(A)
    [N,M,F,T] = A.shape
    
    if crange is None:
        crange = [np.min(A), np.max(A)]
    
    if time_range is None:
        t0 = 0
        t1 = T
    else:
        t0, t1 = time_range
    
    f0, f1 = fs/2, 0
    extent = [t0, t1, f0, f1]
    
    ymin = max(freq_range[0], f1)
    ymax = min(freq_range[1], f0)
    
    if fig is None:
        fig = plt.figure()
    
    axes = []
    for n in range(N):
        arow = []
        for m in range(M):
            ax = fig.add_subplot(N, M, m+n*M+1)
            arow.append(ax)
            
            if n == m:
                if topo:
                    plot_topo(ax, topo, topomaps[m])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_frame_on(False)
            else:
                ax.imshow(A[n,m,:,:], vmin=crange[0], vmax=crange[1], aspect='auto', extent=extent)
                ax.invert_yaxis()
        axes.append(arow)
        
    for n in range(N):
        for m in range(M):
            if n == m:
                pass
            else:
                axes[n][m].xaxis.set_major_locator(MaxNLocator(max(1,9-N)))
                axes[n][m].yaxis.set_major_locator(MaxNLocator(max(1,7-M)))
                axes[n][m].set_ylim(ymin, ymax)
                if 0 < n < N-1:
                    axes[n][m].set_xticks([])
                if 0 < m < M-1:
                    axes[n][m].set_yticks([])                    
        axes[n][0].yaxis.tick_left()
        axes[n][-1].yaxis.tick_right()
        
    for m in range(M):
        axes[0][m].xaxis.tick_top()
        axes[-1][m].xaxis.tick_bottom()
        
    fig.text(0.5, 0.025, 'time', horizontalalignment='center')
    fig.text(0.05, 0.5, 'frequency', horizontalalignment='center', rotation='vertical')
    
    return fig
    