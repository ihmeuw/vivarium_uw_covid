import numpy as np, matplotlib.pyplot as plt, pandas as pd

def plot_results(df_count_list, ymax=0, title='', metric='new_infections'):
    plt.figure(figsize=(11, 4.25), dpi=120)

    s_list = []
    for df in df_count_list:
        
        if metric == 'new_infections':
            s = df.new_infections
        elif metric == 'cumulative_infections':
            s = df.new_infections.cumsum()
            
        else:
            assert 0, f'metric "{metric}" not recognized'
            
        s.plot(color='k', alpha=.25)
        s_list.append(s)

    s = pd.Series(np.median(s_list, axis=0), index=s.index)
    s.plot(color='k', linewidth=3, linestyle='-')
    s = pd.Series(np.percentile(s_list, 2.5, axis=0), index=s.index)
    s.plot(color='k', linewidth=2, linestyle='--')
    s = pd.Series(np.percentile(s_list, 97.5, axis=0), index=s.index)
    s.plot(color='k', linewidth=2, linestyle='--')
        
    plt.ylabel(f"Number of {metric.replace('_', ' ')}")
               
    plt.grid()
    if not ymax:
        ymax = pd.Series(np.median(s_list, axis=0)).dropna().max() * 1.25

    # UW Fall 2020 Quarter starts 2020-09-30 and ends 2020-12-11 (final exam week ends 2020-12-18)
    # https://www.washington.edu/students/reg/2021cal.html
    # TODO: draw a line for when it ends, and note the metric values at this point
    

    plt.axis(ymin=0, ymax=ymax)
    plt.title(title)
