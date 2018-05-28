from pylab import *

class benchmark_actf_derivs:

    def __init__(self, filename, label):
        self.label = label
        self.data = {};

        bnew = True
        with open(filename) as bmfile:
            for line in bmfile:

                lsplit = line.split()

                if len(lsplit) < 5:
                    continue

                if lsplit[0] == 'ACTF':
                    if not bnew:
                        self.data[actf_name] = actf_data # store previous actf's data

                    actf_name = lsplit[10]
                    actf_data = {}
                    bnew = False
                    continue

                if len(lsplit) > 5:
                    if lsplit[5] == 'function':
                        new_mode = lsplit[4]
                        if new_mode == 'fad':
                            actf_data[mode] = mode_data # store individual mode data
                        mode = new_mode
                        mode_data = {}
                        continue

                if lsplit[0][0:2] == 'f:' or lsplit[0][0:2] == 'f+':
                    mode_data[lsplit[0][:-1]] = (float(lsplit[1]), float(lsplit[3]))

                if lsplit[0] == 'f+d1+d2+d3:' and mode=='fad':
                    actf_data[mode] = mode_data # store fad mode data

        self.data[actf_name] = actf_data # store last actf's data

def plot_compare_actfs(benchmark_list, **kwargs):

    nbm = len(benchmark_list)

    fig = figure()
    itp = 0
    for benchmark in benchmark_list:
        for itm, mode in enumerate(['individual', 'fad']):
            itp+=1
            ax = fig.add_subplot(nbm, 2, itp)
            ax.set_title(benchmark.label + ' version, ' + mode + ' function calls')
            ax.set_ylabel('Time per evaluation [ns]')
            for actf in benchmark.data.keys():
                ax.errorbar(benchmark.data[actf][mode].keys(), [v[0] for v in benchmark.data[actf][mode].values()], xerr=None, yerr=[v[1] for v in benchmark.data[actf][mode].values()], **kwargs)
            ax.legend(benchmark.data.keys())
    fig.tight_layout()
    return fig


benchmark_new = benchmark_actf_derivs('benchmark_new.out', 'new')
try:
    benchmark_old = benchmark_actf_derivs('benchmark_old.out', 'old')
    benchmark_list = [benchmark_new, benchmark_old]
except:
    benchmark_list = [benchmark_new]

fig1 = plot_compare_actfs(benchmark_list, fmt='o--')

show()
