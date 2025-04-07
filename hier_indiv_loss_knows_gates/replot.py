import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.manifold import MDS
from itertools import product as pd

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=200)
matplotlib.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

def combine_cmaps(map1, map2):
    cols1 = map1(np.linspace(0.0, 0.5, 128))
    cols2 = map2(np.linspace(0.5, 1.0, 128))
    cols = np.vstack([cols1, np.ones((4,4)), cols2])
    newmap = colors.LinearSegmentedColormap.from_list('new_colormap', cols)
    return newmap

def load_losses():
    names = ['uniform','binomial','habit']
    data = []
    num_runs = 6
    for name in names:
        for run_idx in range(num_runs):
            data.append(np.loadtxt('n_runs/'+str(run_idx) + '_' + name + '_train_losses.txt'))
    return np.array(data).reshape(len(names),num_runs,data[0].shape[0]), names

def load_mds():
    data = np.loadtxt('n_runs/10_binomial_mds.txt')
    data_reshape = np.array(data).reshape(num_samples, 8*3, 100)
    return data_reshape

def plot_outputs(output, cmap, file_name, vmin=-1, vmax=1):
    plt.imshow(output, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    plt.grid()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(file_name, dpi=200, bbox_inches='tight')
    plt.close()

def plot_losses(losses, labels, colors, linestyles, file_name='losses.pdf', add_legend=True):
    mean_losses = np.mean(losses, axis=1)
    std_losses = np.std(losses, axis=1)
    for i in range(len(mean_losses)):
        plt.plot(mean_losses[i,:8000], color=colors[i], label=labels[i], linestyle=linestyles[i])
        plt.fill_between(np.arange(8000), mean_losses[i,:8000] - std_losses[i,:8000],\
                                          mean_losses[i,:8000] + std_losses[i,:8000], color=colors[i], alpha=0.3)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    #plt.axvline(500, color='black')
    plt.ylabel("Quadratic Loss")
    plt.xlabel("Epoch number")
    if add_legend:
        plt.legend()
    plt.tight_layout()
    plt.savefig(file_name, dpi=400, bbox_inches='tight')
    plt.close()

def plot_mds(hidden_units, jump_size, label, color):
    num_epochs = hidden_units.shape[0]
    num_data = hidden_units.shape[1]
    num_hidden = hidden_units.shape[2]
    mds_object = MDS(normalized_stress='auto', random_state=42)
    c = np.linspace(0,1,int(np.ceil(num_epochs/jump_size))+2)[:-1][1:].astype(np.float32)
    mds = mds_object.fit_transform(hidden_units[::jump_size].reshape((int(np.ceil(num_epochs/jump_size)))*num_data,num_hidden))
    mds = mds.reshape((int(np.ceil(num_epochs/jump_size))),num_data,2)
    for k in range(mds.shape[1]):
        plt.scatter(mds[:,k,0], mds[:,k,1], c=c, s=2, marker='o', cmap=color)
        plt.text(mds[-1,k,0] + (mds[-1,k,0] - mds[-2,k,0]), mds[-1,k,1] + (mds[-1,k,1] - mds[-2,k,1]), str(k))
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    #plt.tight_layout()
    plt.savefig('plots/binomial_mds.pdf',dpi=200)
    plt.close()

if __name__ == "__main__":

    plt.style.use('ggplot')
    matplotlib.rcParams.update({'font.size': 14})
    cols1 = plt.cm.PiYG_r(np.linspace(0.7, 1.0, 256))
    new_cmap_pink = colors.LinearSegmentedColormap.from_list('new_colormap_pink', cols1)
    cols2 = plt.cm.BrBG(np.linspace(0.7, 1.0, 256))
    new_cmap_blue = colors.LinearSegmentedColormap.from_list('new_colormap_blue', cols2)

    loss = True
    mds = False

    if loss:
        losses, file_names = load_losses()
        labels = file_names
        colors = ['deeppink','blue','green']
        linestyles = ['-','-','-']
        plot_losses(losses, labels, colors, linestyles, file_name='plots/losses.pdf', add_legend=True)

    if mds:
        sample_size = 10
        jump_size = 10 #10
        num_samples = int(10000/sample_size)+1
        hidden_units = load_mds()
        label = 'meta-GDLN'
        plot_mds(hidden_units, jump_size, label, new_cmap_blue) 
