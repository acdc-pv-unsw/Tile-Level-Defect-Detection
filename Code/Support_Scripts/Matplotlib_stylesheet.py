import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn-paper')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] ='STIXGeneral'
mpl.rcParams['mathtext.default'] = 'rm'
mpl.rcParams['mathtext.fallback'] = 'cm'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.labelweight'] = 'normal'
mpl.rcParams['axes.grid.which']='both'
mpl.rcParams['grid.linewidth']= 0
mpl.rcParams['axes.xmargin']=0.05
mpl.rcParams['axes.ymargin']=0.05
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['legend.title_fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 18
#mpl.rcParams['figure.figsize'] = (16.18,10)
mpl.rcParams['figure.figsize'] = (16.18,12)
mpl.rcParams['figure.autolayout'] = False
mpl.rcParams['image.cmap'] = "Paired"
mpl.rcParams['figure.dpi'] = 75
mpl.rcParams['savefig.dpi'] = 150
mpl.rcParams['errorbar.capsize'] = 3
mpl.rcParams['axes.prop_cycle'] = plt.cycler(color = plt.cm.Set1(np.linspace(0.1,0.9,10)))
