from pylab import *

class benchmark_container:

    def __init__(self, filename, label):
        self.label = label
        self.data = {};

        bnew = True
        with open(filename) as bmfile:
            for line in bmfile:

                lsplit = line.split()

                if len(lsplit) < 2:
                    continue

                if lsplit[0] == 'ACTF':
                    if not bnew:
                        self.data[actf_name] = actf_data # store previous actf's data

                    actf_name = lsplit[7]
                    actf_data = {}
                    bnew = False
                    continue

                if lsplit[1] == 'function':
                    new_mode = lsplit[0]
                    if new_mode == 'fad':
                        actf_data[mode] = mode_data # store individual mode data
                    mode = new_mode
                    mode_data = {}
                    continue

                if lsplit[0][0:2] == 'f:' or lsplit[0][0:2] == 'f+':
                    mode_data[lsplit[0][:-1]] = float(lsplit[1])

                if lsplit[0] == 'f+d1+d2+d3:' and mode=='fad':
                    actf_data[mode] = mode_data # store fad mode data

        self.data[actf_name] = actf_data # store last actf's data


mybenchmark = benchmark_container('benchmark_new.out', 'new')
print(mybenchmark.label)
print(mybenchmark.data)
print(mybenchmark.data['lgs']['fad'].keys(), mybenchmark.data['lgs']['fad'].values())


actfs = ['lgs', 'gss', 'tans', 'id_', 'sin', 'relu', 'selu']

figure(1)
for actf in actfs:
    plot(mybenchmark.data[actf]['individual'].keys(), mybenchmark.data[actf]['individual'].values(), 'o--')
legend(actfs)

figure(2)
for actf in actfs:
    plot(mybenchmark.data[actf]['fad'].keys(), mybenchmark.data[actf]['fad'].values(), 'o--')
legend(actfs)

tight_layout()
show()
