import numpy as np
import torch as t
import math



import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import math
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def make_fourier_samples(actions, wavevectors, phases, rand, n_cos=100):
    freq = actions @ wavevectors
    y = np.cos(freq + phases)
    obs = math.sqrt(2)*y @ rand/math.sqrt(n_cos)
    return obs

class animate_fourier:
    def __init__(self):
        self._data_dict_list = None
        self.xmin, self.xmax = 0,1

    def contour(self, ax, axcolor, wavevectors, phases, rand, lengthscale=None):
        if lengthscale is None:
            lengthscale = self.args.lengthscale
        # ax.set_xticks([0,1])
        # ax.set_yticks([0,1])

        xmin, xmax = self.xmin, self.xmax
        ymin, ymax = xmin, xmax

        ax.set_xticks([xmin,xmax])
        ax.set_yticks([xmin,xmax])
        levels = [-2,-1.5,-1,-.5,0,.5,1,1.5,2]

        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])

        f = make_fourier_samples(positions.T, wavevectors, phases, rand).reshape(xx.shape)
        fmax, fmin = np.max(f), np.min(f)
        levels = np.linspace(fmin,fmax, 10)
        cmap = plt.cm.get_cmap('Blues_r')
        
        fmax, fmin = np.max(f), np.min(f)
        levels = np.linspace(fmin,fmax, 20)

        cs = ax.contourf(xx, yy, f, levels, cmap=cmap, extend='max', origin='lower')

        vmin, vmax = levels[0],levels[-1]
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        cbar = mpl.colorbar.ColorbarBase(axcolor, cmap=cmap, orientation = 'vertical', norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        cbar.set_ticks([levels[0],levels[-1]])


    def plot(self, fname=None):
        xmin, xmax = self.xmin, self.xmax

        actions_list = []
        wavevectors_list = []
        phases_list = []
        l_list = []
        rand_list = []

        for data_dict in self._data_dict_list:
            l_list.append(data_dict['l'])
            actions_list.append(data_dict['action'])
            # actions_list.append(np.clip(data_dict['action'],0,1))
            # actions_list.append(1 / (1 + np.exp(-5*data_dict['action'])))
            # actions_list.append(np.tanh(data_dict['action']))
            wavevectors_list.append(data_dict['wavevectors']/data_dict['l'])
            phases_list.append(data_dict['phases'])
            rand_list.append(data_dict['rand'])
        
        fig = plt.figure(figsize=(4.5,3.2))
        bottom = 0.2
        top = 0.85
        left = 0.1
        right = 0.85
        gs = GridSpec(1, 1, figure=fig,
                    wspace=0.5, hspace=0.2, left=left, right=right, bottom=bottom, top=top
                    )
        ax_list = [fig.add_subplot(gs[0, i]) for i in range(1)]

        # ax_list[0].set_xlim([-0,1])
        # ax_list[0].set_ylim([-0,1])
        # ax_list[0].set_xticks([0,1])
        # ax_list[0].set_yticks([0,1])

        ax_list[0].set_xlim([xmin,xmax])
        ax_list[0].set_ylim([xmin,xmax])
        ax_list[0].set_xticks([xmin,xmax])
        ax_list[0].set_yticks([xmin,xmax])

        ax_list[0].set_xlabel('x1')
        ax_list[0].set_ylabel('x2')
        ax_list[0].set_title('GP sample')

        ax0_color = inset_axes(ax_list[0],
                    width="5%",  # width = 5% of parent_bbox width
                    height="100%",  
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0., 1, 1),
                    bbox_transform=ax_list[0].transAxes,
                    borderpad=0,
                    )


        def animate(i):
            print(f'plotting {i}')


            ax_list[0].clear()
            ax0_color.clear()

            ax_list[0].set_xlim([xmin,xmax])
            ax_list[0].set_ylim([xmin,xmax])
            ax_list[0].set_xticks([xmin,xmax])
            ax_list[0].set_yticks([xmin,xmax])


            ax_list[0].set_xlabel('x1')
            ax_list[0].set_ylabel('x2')
            ax_list[0].set_title(f'GP sample {l_list[i]:.2f}')

            self.contour(ax_list[0],ax0_color, wavevectors_list[i],phases_list[i], rand_list[i],  lengthscale=l_list[i])


            ax_list[0].scatter(actions_list[i][:,0],actions_list[i][:,1], marker='.', color='salmon',s=100)


        ani = FuncAnimation(fig, animate, frames=len(self._data_dict_list), interval=50)
        if fname is None:
            ani.save(f"/user/home/ad20999/transformers-metarl/plots/{fname}.gif", dpi=300, writer=PillowWriter(fps=5))
        elif fname is not None:
            ani.save(f"/user/home/ad20999/transformers-metarl/plots/{fname}.gif", dpi=300, writer=PillowWriter(fps=5))
        