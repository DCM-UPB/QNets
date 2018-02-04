from pylab import * 

gauss_a = 1.0
gauss_b = 0

names = ['v']
files = []
for i in range(len(names)):
    files.append(names[i]+'.txt')
    names[i] = 'nn ' + names[i]

x_list = []
nn_list = []

for file in files:

    x = []
    nn = []
    for line in open(file):
        llist = line.split('   ')
        x.append(float(llist[0]))
        nn.append(float(llist[1]))

    x = array(x)
    nn = array(nn)

    x_list.append(x)
    nn_list.append(nn)

gauss = []
for x in x_list[0]:
    gauss.append(exp(-gauss_a * (x - gauss_b)**2))
gauss = array(gauss)

diff_list = []
for nn in nn_list:
    diff = 0
    itv = 0
    for val in nn:
        diff += (val-gauss[itv])**2
        itv+=1
    diff /= itv
    diff_list.append(diff)

figure(1)
plot(x_list[0], gauss)
for i in range(len(x_list)):
    plot(x_list[i], nn_list[i])

xlabel('x')
ylabel('f(x)')
title('NN Fitting Example')

for i in range(len(names)):
    names[i] += ' MD2: ' + "{:.5f}".format(diff_list[i])
legend(['gauss'] + names)
show()
#savefig('plot.pdf')


