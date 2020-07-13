import numpy as np, matplotlib.pyplot as plt, pandas as pd

def hop_plot(df_count_list, ymax=0, title='', metric='new_infections'):
    from IPython.display import clear_output
    from time import sleep
    
    try:
        while True:
            plot_results(df_count_list[:100], ymax, title, metric)
            df = df_count_list[np.random.choice(range(len(df_count_list)))]
            if metric == 'new_infections':
                s = df.n_new_infections
            elif metric == 'cumulative_infections':
                s = df.n_new_infections.cumsum()
            else:
                assert 0, f'metric "{metric}" not recognized'
            s.plot(color='C0', alpha=1, linewidth=5)

            clear_output(wait=True)
            plt.show()
    except KeyboardInterrupt:
        pass

def plot_results(df_count_list, ymax=0, title='', metric='new_infections', end_dates=['2020-12-11']):
    plt.figure(figsize=(11, 4.25), dpi=120)

    s_list = []
    for df in df_count_list:
        
        if metric == 'new_infections':
            s = df.n_new_infections
        elif metric == 'cumulative_infections':
            s = df.n_new_infections.cumsum()
        elif metric == 'new_isolations':
            s = df.n_new_isolations
            
        else:
            assert 0, f'metric "{metric}" not recognized'
            
        s.plot(color='k', alpha=.25)
        s_list.append(s)

    s = pd.Series(np.median(s_list, axis=0), index=s.index)
    s.plot(color='k', linewidth=3, linestyle='-')
    s_median = s
    
    s = pd.Series(np.percentile(s_list, 2.5, axis=0), index=s.index)
    s.plot(color='k', linewidth=2, linestyle='--')
    s_lb = s

    s = pd.Series(np.percentile(s_list, 97.5, axis=0), index=s.index)
    s.plot(color='k', linewidth=2, linestyle='--')
    s_ub = s
        
    plt.ylabel(f"Number of {metric.replace('_', ' ')}")
               
    plt.grid()
    if not ymax:
        ymax = pd.Series(np.median(s_list, axis=0)).dropna().max() * 1.25

    # UW Fall 2020 Quarter starts 2020-09-30 and ends 2020-12-11 (final exam week ends 2020-12-18)
    # https://www.washington.edu/students/reg/2021cal.html
    # TODO: draw a line for when it ends, and note the metric values at this point
    #plt.plot([end_date, end_date], [0, ymax], 'k:')
    summary_result_str = ''
    for end_date in end_dates:
        end_date = pd.Timestamp(end_date)
    
        summary_result_str += f"\n{metric.replace('_', ' ').capitalize()} on "
        summary_result_str += f"{end_date.strftime('%D')}:\n    {s_median[end_date]:,.0f} (95% UI {s_lb[end_date]:,.0f} to {s_ub[end_date]:,.0f})\n"
    print(summary_result_str)
    plt.text(s.index[0], ymax, summary_result_str, ha='left', va='top')
    plt.axis(ymin=0, ymax=ymax)
    plt.title(title)


def plot_medians_over_time(results, legend_title):
    plt.figure(figsize=(11, 4.25), dpi=120)

    for val, df_list in results.items():
        cumsum_mean = np.median([t.n_new_infections.cumsum() for t in df_list], axis=0)
        plt.plot(df_list[0].index, cumsum_mean, label=val)
    # plt.xticks(rotation=90)
    plt.legend(loc=(1.01, .01), title=legend_title)
    plt.ylabel(f"Median cumulative infections")
    plt.grid()
#     plt.semilogy()


def plot_medians_at_end_of_quarter(results, close_date, xlabel):
    plt.figure(figsize=(11, 4.25), dpi=120)

    xx, yy, yy_lb, yy_ub = [], [], [], []
    for val, df_list in results.items():
        cum_cases = [t.n_new_infections.cumsum()[close_date] for t in df_list]
        xx.append(val)
        yy.append(np.median(cum_cases))
        yy_lb.append(np.median(cum_cases) - np.percentile(cum_cases, 2.5))
        yy_ub.append(np.percentile(cum_cases, 97.5) - np.median(cum_cases))
        
        
    plt.errorbar(xx, yy, yerr=[yy_lb, yy_ub], marker='o', linestyle='-')
    plt.ylabel(f"Cumulative infections on {close_date.strftime('%D')}")
    plt.xlabel(xlabel)
    plt.grid()
    plt.axis(ymin=0)


def plot_w_ui(results, testing, title, ymax=0):
    plot_results(results[testing], ymax=ymax,
                 title=f'{title} {testing}',
                 metric='cumulative_infections')

