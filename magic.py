import time

def display_est_time_loop(tot_toc, curr_ix, tot_iter, prefix=''):
    if curr_ix == tot_iter:
        neat_time = time.strftime('%H:%M:%S', time.gmtime(tot_toc))
        print("\r" + prefix + 'Total elapsed time (HH:MM:SS): ' + neat_time, end='')
        print('')
    else:
        avg_toc = tot_toc / curr_ix
        estimated_time_hours = (avg_toc * (tot_iter - curr_ix))
        neat_time = time.strftime('%H:%M:%S', time.gmtime(estimated_time_hours))
        perc = str(round(curr_ix*100/tot_iter))
        print('\r' + prefix + 'Estimated time (HH:MM:SS): ' + neat_time + ' ' + perc + '%', end='')
    return tot_toc