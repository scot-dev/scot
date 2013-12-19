# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Graphical output with matplotlib """

import numpy as np
from . import var

try:
    #noinspection PyPep8Naming
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    _have_pyplot = True
except ImportError:
    plt, MaxNLocator = None, None
    _have_pyplot = False


def show_plots():
    plt.show()


def prepare_topoplots(topo, values):
    values = np.atleast_2d(values)

    topomaps = []

    for i in range(values.shape[0]):
        topo.set_values(values[i, :])
        topo.create_map()
        topomaps.append(topo.get_map())

    return topomaps


def plot_topo(axis, topo, topomap, crange=None, offset=(0,0)):
    topo.set_map(topomap)
    h = topo.plot_map(axis, crange=crange, offset=offset)
    topo.plot_locations(axis, offset=offset)
    topo.plot_head(axis, offset=offset)
    return h


def plot_sources(topo, mixmaps, unmixmaps, global_scale=None, fig=None):
    """ global_scale:
           None - scales each topo individually
           1-99 - percentile of maximum of all plots
    """
    if not _have_pyplot:
        raise ImportError("matplotlib.pyplot is required for plotting")

    urange, mrange = None, None

    m = len(mixmaps)

    if global_scale:
        tmp = np.asarray(unmixmaps)
        tmp = tmp[np.logical_not(np.isnan(tmp))]
        umax = np.percentile(np.abs(tmp), global_scale)
        umin = -umax
        urange = [umin, umax]

        tmp = np.asarray(mixmaps)
        tmp = tmp[np.logical_not(np.isnan(tmp))]
        mmax = np.percentile(np.abs(tmp), global_scale)
        mmin = -mmax
        mrange = [mmin, mmax]

    y = np.floor(np.sqrt(m * 3 / 4))
    x = np.ceil(m / y)

    if fig is None:
        fig = plt.figure()

    axes = []
    for i in range(m):
        axes.append(fig.add_subplot(2 * y, x, i + 1))
        plot_topo(axes[-1], topo, unmixmaps[i], crange=urange)
        axes[-1].set_title(str(i))

        axes.append(fig.add_subplot(2 * y, x, m + i + 1))
        plot_topo(axes[-1], topo, mixmaps[i], crange=mrange)
        axes[-1].set_title(str(i))

    for a in axes:
        a.set_yticks([])
        a.set_xticks([])
        a.set_frame_on(False)

    axes[0].set_ylabel('Unmixing weights')
    axes[1].set_ylabel('Scalp projections')

    return fig


def plot_connectivity_spectrum(a, fs=2, freq_range=(-np.inf, np.inf), topo=None, topomaps=None, fig=None):
    a = np.atleast_3d(a)
    [n, m, f] = a.shape
    freq = np.linspace(0, fs / 2, f)

    lowest, highest = np.inf, -np.inf
    left = max(freq_range[0], freq[0])
    right = min(freq_range[1], freq[-1])

    if fig is None:
        fig = plt.figure()

    axes = []
    for i in range(n):
        arow = []
        for j in range(m):
            ax = fig.add_subplot(n, m, j + i * m + 1)
            arow.append(ax)

            if i == j:
                if topo:
                    plot_topo(ax, topo, topomaps[j])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_frame_on(False)
            else:
                ax.plot(freq, a[i, j, :])
                lowest = min(lowest, np.min(a[i, j, :]))
                highest = max(highest, np.max(a[i, j, :]))
                ax.set_xlim(0, fs / 2)
        axes.append(arow)

    for i in range(n):
        for j in range(m):
            if i == j:
                pass
            else:
                axes[i][j].xaxis.set_major_locator(MaxNLocator(max(1, 7 - n)))
                axes[i][j].yaxis.set_major_locator(MaxNLocator(max(1, 7 - m)))
                axes[i][j].set_ylim(lowest, highest)
                axes[i][j].set_xlim(left, right)
                if 0 < i < n - 1:
                    axes[i][j].set_xticks([])
                if 0 < j < m - 1:
                    axes[i][j].set_yticks([])
        axes[i][0].yaxis.tick_left()
        axes[i][-1].yaxis.tick_right()

    for j in range(m):
        axes[0][j].xaxis.tick_top()
        axes[-1][j].xaxis.tick_bottom()

    fig.text(0.5, 0.025, 'frequency (Hz)', horizontalalignment='center')
    fig.text(0.05, 0.5, 'magnitude', horizontalalignment='center', rotation='vertical')

    return fig


def plot_connectivity_timespectrum(a, fs=2, crange=None, freq_range=(-np.inf, np.inf), time_range=None, topo=None,
                                   topomaps=None, fig=None):
    a = np.asarray(a)
    [n, m, _, t] = a.shape

    if crange is None:
        crange = [np.min(a), np.max(a)]

    if time_range is None:
        t0 = 0
        t1 = t
    else:
        t0, t1 = time_range

    f0, f1 = fs / 2, 0
    extent = [t0, t1, f0, f1]

    ymin = max(freq_range[0], f1)
    ymax = min(freq_range[1], f0)

    if fig is None:
        fig = plt.figure()

    axes = []
    for i in range(n):
        arow = []
        for j in range(m):
            ax = fig.add_subplot(n, m, j + i * m + 1)
            arow.append(ax)

            if i == j:
                if topo:
                    plot_topo(ax, topo, topomaps[j])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_frame_on(False)
            else:
                ax.imshow(a[i, j, :, :], vmin=crange[0], vmax=crange[1], aspect='auto', extent=extent)
                ax.invert_yaxis()
        axes.append(arow)

    for i in range(n):
        for j in range(m):
            if i == j:
                pass
            else:
                axes[i][j].xaxis.set_major_locator(MaxNLocator(max(1, 9 - n)))
                axes[i][j].yaxis.set_major_locator(MaxNLocator(max(1, 7 - m)))
                axes[i][j].set_ylim(ymin, ymax)
                if 0 < i < n - 1:
                    axes[i][j].set_xticks([])
                if 0 < j < m - 1:
                    axes[i][j].set_yticks([])
        axes[i][0].yaxis.tick_left()
        axes[i][-1].yaxis.tick_right()

    for j in range(m):
        axes[0][j].xaxis.tick_top()
        axes[-1][j].xaxis.tick_bottom()

    fig.text(0.5, 0.025, 'time (s)', horizontalalignment='center')
    fig.text(0.05, 0.5, 'frequency (Hz)' , horizontalalignment='center', rotation='vertical')

    return fig


def plot_circular(widths, colors, curviness=0.2, mask=True, topo=None, topomaps=None, axes=None, order=None):
    colors = np.asarray(colors)
    widths = np.asarray(widths)
    mask = np.asarray(mask)

    colors = np.maximum(colors, 0)
    colors = np.minimum(colors, 1)

    if len(widths.shape) > 2:
        [n, m] = widths.shape
    elif len(colors.shape) > 3:
        [n, m, c] = widths.shape
    elif len(mask.shape) > 2:
        [n, m] = mask.shape
    else:
        n = len(topomaps)
        m = n

    if not order:
        order = list(range(n))

    #a = np.asarray(a)
    #[n, m] = a.shape

    assert(n == m)

    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111)
    axes.set_yticks([])
    axes.set_xticks([])
    axes.set_frame_on(False)

    if len(colors.shape) < 3:
        colors = np.tile(colors, (n,n,1))

    if len(widths.shape) < 2:
        widths = np.tile(widths, (n,n))

    if len(mask.shape) < 2:
        mask = np.tile(mask, (n,n))
    np.fill_diagonal(mask, False)

    if topo:
        r = 1.25 * topo.head_radius / (np.sin(np.pi/n))
    else:
        r = 1

    for i in range(n):
        if topo:
            o = (r*np.sin(i*2*np.pi/n), r*np.cos(i*2*np.pi/n))
            plot_topo(axes, topo, topomaps[order[i]], offset=o)

    for i in range(n):
        for j in range(n):
            if not mask[order[i], order[j]]:
                continue
            a0 = j*2*np.pi/n
            a1 = i*2*np.pi/n

            x0, y0 = r*np.sin(a0), r*np.cos(a0)
            x1, y1 = r*np.sin(a1), r*np.cos(a1)

            ex = (x0 + x1) / 2
            ey = (y0 + y1) / 2
            en = np.sqrt(ex**2 + ey**2)

            if en < 1e-10:
                en = 0
                ex = y0 / r
                ey = -x0 / r
                w = -r
            else:
                ex /= en
                ey /= en
                w = np.sqrt((x1-x0)**2 + (y1-y0)**2) / 2

                if x0*y1-y0*x1 < 0:
                    w = -w

            d = en*(1-curviness)
            h = en-d

            t = np.linspace(-1, 1, 100)

            dist = (t**2+2*t+1)*w**2 + (t**4-2*t**2+1)*h**2

            tmask1 = dist >= (1.4*topo.head_radius)**2
            tmask2 = dist >= (1.2*topo.head_radius)**2
            tmask = np.logical_and(tmask1, tmask2[::-1])
            t = t[tmask]

            x = (h*t*t+d)*ex - w*t*ey
            y = (h*t*t+d)*ey + w*t*ex

            # Arrow Head
            s = np.sqrt((x[-2] - x[-1])**2 + (y[-2] - y[-1])**2)

            width = widths[order[i], order[j]]

            x1 = 0.1*width*(x[-2] - x[-1] + y[-2] - y[-1])/s + x[-1]
            y1 = 0.1*width*(y[-2] - y[-1] - x[-2] + x[-1])/s + y[-1]

            x2 = 0.1*width*(x[-2] - x[-1] - y[-2] + y[-1])/s + x[-1]
            y2 = 0.1*width*(y[-2] - y[-1] + x[-2] - x[-1])/s + y[-1]

            x = np.concatenate([x, [x1, x[-1], x2]])
            y = np.concatenate([y, [y1, y[-1], y2]])
            axes.plot(x, y, lw=width, color=colors[order[i], order[j]], solid_capstyle='round', solid_joinstyle='round')

    return axes


def plot_whiteness(var, h, repeats=1000, axis=None):
    pr, q0, q = var.test_whiteness(h, repeats, True)

    if axis is None:
        axis = plt.gca()

    pdf, _, _ = axis.hist(q0, 30, normed=True, label='surrogate distribution')
    axis.plot([q,q], [0,np.max(pdf)], 'r-', label='fitted model')

    #df = m*m*(h-p)
    #x = np.linspace(np.min(q0)*0.0, np.max(q0)*2.0, 100)
    #y = sp.stats.chi2.pdf(x, df)
    #hc = axis.plot(x, y, label='chi-squared distribution (df=%i)' % df)

    axis.set_title('significance: p = %f'%pr)
    axis.set_xlabel('Li-McLeod statistic (Q)')
    axis.set_ylabel('probability')

    axis.legend()

    return pr
