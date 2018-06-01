from pylab import *

class benchmark_actf_ffprop:

    def __init__(self, filename, label):
        self.label = label
        self.data = {};

        bnew = True
        with open(filename) as bmfile:
            for line in bmfile:

                lsplit = line.split()

                if len(lsplit) < 5:
                    continue

                if lsplit[0] == 'FFPropagate':
                    if not bnew:
                        self.data[actf_name] = actf_data # store previous actf's data

                    actf_name = lsplit[9]
                    actf_data = {}
                    bnew = False
                    continue

                if lsplit[0][0:2] == 'f:' or lsplit[0][0:2] == 'f+':
                    actf_data[lsplit[0][:-1]] = (float(lsplit[1]), float(lsplit[3]))

        self.data[actf_name] = actf_data # store last actf's data


def plot_compare_actfs(benchmark_list, **kwargs):
    nbm = len(benchmark_list)
    xlabels = benchmark_list[0].data['lgs'].keys()

    fig = figure()
    fig.suptitle('FFPropagate benchmark, comparing all actfs',fontsize=14)

    itp = 0
    for benchmark in benchmark_list:

        itp+=1
        ax = fig.add_subplot(nbm, 1, itp)
        for actf in benchmark.data.keys():
            values = [v[0] for v in benchmark.data[actf].values()]
            errors = [v[1] for v in benchmark.data[actf].values()]
            ax.errorbar(xlabels, values, xerr=None, yerr=errors, **kwargs)

        ax.set_yscale('log')
        ax.set_title(benchmark.label + ' version')
        ax.set_ylabel('Time per propagation [$\mu s$]')
        ax.legend(benchmark.data.keys())

    return fig


def plot_compare_runs(benchmark_list, actf_list, width = 0.75, **kwargs):
    nbm = len(benchmark_list)
    bwidth = width/float(nbm)
    nactf = len(actf_list)
    if nbm > 1: ind = arange(len(benchmark_list[0].data[actf_list[0]]))
    else: ind = arange(len(benchmark_list[0].data[actf_list[0]])) + 0.5*bwidth
    xlabels = benchmark_list[0].data[actf_list[0]].keys()

    fig = figure()
    fig.suptitle('FFPropagate benchmark, comparing versions for selected actfs',fontsize=14)

    itp = 0
    for actf in actf_list:

            itp+=1
            ax = fig.add_subplot(nactf, 1, itp)
            scales = array([100./v[0] for v in benchmark_list[0].data[actf].values()]) # we will normalize data to the first benchmark's results
            for itb, benchmark in enumerate(benchmark_list):
                values = array([v[0] for v in benchmark.data[actf].values()])*scales
                errors = array([v[1] for v in benchmark.data[actf].values()])*scales
                ax.bar(ind + itb*bwidth, values, bwidth, yerr=errors, **kwargs)

            ax.set_title(actf + ' actf')
            ax.set_ylabel('Time per propagation [%]')
            ax.set_xticks(ind + 0.5*(nbm-1)*bwidth)
            ax.set_xticklabels(xlabels)
            ax.legend([benchmark.label for benchmark in benchmark_list])

    return fig

# Script

benchmark_list = []
for benchmark_file in sys.argv[1:]:
    try:
        benchmark = benchmark_actf_ffprop(benchmark_file, benchmark_file.split('_')[1].split('.')[0])
        benchmark_list.append(benchmark)
    except:
        print("Warning: Couldn't load benchmark file " + benchmark_file + "!")

if len(benchmark_list)<1: print("Error: Not even one benchmark loaded!")
else:
    fig1 = plot_compare_actfs(benchmark_list, fmt='o--')
    if len(benchmark_list)>1: fig2 = plot_compare_runs(benchmark_list, ['tans', 'gss', 'relu'])

show()
