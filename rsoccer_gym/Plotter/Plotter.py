import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing as mp

class RealTimePlotter:
    '''
    Class for plotting real-time data on a parallel process.
    '''
    def __init__(self,
                 n_plots,
                 max_size,
                 titles):
        '''
        Creates Real Time Plotter object. Uses same "x" for all plots.

        Parameters
        ----------
        n_plots : int
            Number of plots to be displayed

        masx_size : int
            Maximum plot size, usually number of steps from simulation
            
        '''
        self.n_plots = n_plots
        self.max_size = max_size
        self.ax_titles = titles

        self.xs = mp.Queue(maxsize=max_size)
        self.ys = mp.Queue(maxsize=n_plots*max_size)
        self.side_process = mp.Process(target=self.process_plot, args=(self.xs, self.ys))
        self.side_process.start()

    def set_axis(self, ax, ax_nr):
        ax.clear()
        if ax_nr==0: ax.set_ylim(0, 5)
        if ax_nr==1: ax.set_ylim(0, 1)
        if ax_nr==2: ax.set_ylim(0, 1)
        ax.set_title(self.ax_titles[ax_nr])

    def process_plot(self, xs, ys):
        # PLOT CONFIGS:
        fig = plt.figure(figsize=(6.75, 9.3))
        ax = fig.subplots(nrows=self.n_plots, ncols=1)
        fig.subplots_adjust(left=0.1, 
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        list_x = []
        list_ys = []
        for i in range(0, self.n_plots):
            list_ys.append([])

        def update(*args, **kwargs):
            list_x = args[1]
            list_ys = args[2]
            _x = xs.get()
            _y = ys.get()
            list_x.append(_x)
            for i in range(0, self.n_plots):
                list_ys[i].append(_y[i])
                self.set_axis(ax[i], i)
                ax[i].plot(list_x, list_ys[i])

        _ = animation.FuncAnimation(fig, update, fargs=(list_x, list_ys), interval=1)
        plt.show()

    def kill_process(self):
        self.side_process.terminate()
        self.side_process.join()

    def add_data(self, new_x, new_ys):
        '''
        Sends new data to plotter process.

        new_x: new step to x axis
        new_ys: a list containing data for all y's axis
        '''
        if len(new_ys)==self.n_plots:
            self.xs.put(new_x)
            self.ys.put(new_ys)
    