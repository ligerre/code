import matplotlib.pyplot as plt

def make_plot(values,labels,name):
    for i in range(len(values)):
        plt.plot(values[i],label = labels[i])
    plt.yscale('log')
    plt.xlabel(u'Iteration')
    plt.ylabel(name)
    plt.legend()