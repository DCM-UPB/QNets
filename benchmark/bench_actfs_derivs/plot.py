from pylab import *
import sys

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
    xlabels = benchmark_list[0].data['LGS']['fad'].keys()

    fig = figure()
    fig.suptitle('Activation function derivative benchmark, comparing all ActF',fontsize=14)

    itp = 0
    for benchmark in benchmark_list:
        for mode in ['individual', 'fad']:

            itp+=1
            ax = fig.add_subplot(nbm, 2, itp)
            for actf in benchmark.data.keys():
                values = [v[0] for v in benchmark.data[actf][mode].values()]
                errors = [v[1] for v in benchmark.data[actf][mode].values()]
                ax.errorbar(xlabels, values, xerr=None, yerr=errors, **kwargs)

            ax.set_title(benchmark.label + ' version, ' + mode + ' function calls')
            if mode=='individual': ax.set_ylabel('Time per eval [ns]')
            ax.legend(benchmark.data.keys())

    return fig


def plot_compare_runs(benchmark_list, actf_list, width = 0.8, **kwargs):
    nbm = len(benchmark_list)-1
    if nbm <= 0: 
        print('Error: Not enough benchmarks for comparison plot.')
        return None

    bwidth = width/float(nbm)
    nactf = len(actf_list)
    if nbm > 1: ind = arange(len(benchmark_list[0].data[actf_list[0]]['fad']), 0, -1)
    else: ind = arange(len(benchmark_list[0].data[actf_list[0]]['fad']), 0, -1) + 0.5*bwidth
    xlabels = benchmark_list[0].data[actf_list[0]]['fad'].keys()

    fig = figure()
    fig.suptitle('Activation function derivative benchmark, comparing against ' + benchmark_list[0].label + ' version',fontsize=14)

    itp = 0
    for ita, actf in enumerate(actf_list):
        for mode in ['individual', 'fad']:

            itp+=1
            ax = fig.add_subplot(nactf, 2, itp)
            scales = array([100./v[0] for v in benchmark_list[0].data[actf][mode].values()]) # we will normalize data to the first benchmark's results
            for itb, benchmark in enumerate(benchmark_list[1:]):
                values = array([v[0] for v in benchmark.data[actf][mode].values()])*scales
                errors = array([v[1] for v in benchmark.data[actf][mode].values()])*scales
                rects = ax.barh(ind - itb*bwidth, values, bwidth, xerr=errors, **kwargs)
                for rect in rects:
                    ax.text(1, rect.get_y() + rect.get_height()/2., '%d' % int(rect.get_width()), ha='left', va='center', fontsize=8)

            ax.set_title(actf + ' actf, ' + mode + ' function calls')
            if ita==len(actf_list)-1: ax.set_xlabel(r'Time per eval (%)')
            ax.set_xlim([0,200])
            ax.set_yticks(ind - 0.5*(nbm-1)*bwidth)
            ax.set_yticklabels(xlabels)
            ax.legend([benchmark.label for benchmark in benchmark_list[1:]])

    return fig


# Script

benchmark_list = []
for benchmark_file in sys.argv[1:]:
    try:
        benchmark = benchmark_actf_derivs(benchmark_file, benchmark_file.split('_')[1].split('.')[0])
        benchmark_list.append(benchmark)
    except:
        print("Warning: Couldn't load benchmark file " + benchmark_file + "!")

if len(benchmark_list)<1: print("Error: Not even one benchmark loaded!")
else:
    fig1 = plot_compare_actfs(benchmark_list, fmt='o--')
    if len(benchmark_list)>1: fig2 = plot_compare_runs(benchmark_list, ['TANS', 'GSS', 'RELU'])

show()
