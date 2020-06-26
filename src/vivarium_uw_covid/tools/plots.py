import numpy as np, matplotlib.pyplot as plt, pandas as pd

def plot_results(df_count_list, ymax=0, title='', metric='new_infections'):
    plt.figure(figsize=(11, 4.25), dpi=120)
    for df in df_count_list:
        
        if metric == 'new_infections':
            s = df.new_infections
        elif metric == 'cumulative_infections':
            s = df.new_infections.cumsum()
            
        else:
            assert 0, f'metric "{metric}" not recognized'
            
        s.plot(color='k', alpha=.25)
        
    plt.ylabel(f"Number of {metric.replace('_', ' ')}")
               
    plt.grid()
    if ymax:
        plt.axis(ymin=0, ymax=ymax)
    plt.title(title)
