import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke


def compatible(A, B):
    print(A)
    a = len(A)
    b = len(B)
    X = []

    Min = min(a, b)
    if not a > Min:
        i = 0
        for item in A:
            X.append(B[i])
            i = i + 1
        return A, X
    else:
        i = 0
        for item in B:
            X.append(A[i])
            i = i + 1
        return X, B

def CDF_Probability(data, T):
    data = np.array(data)
    return (1 - np.sum(data >= T) / data.size)

class LegendHandler(Line2D):
    def __init__(self, color, label, hatch_pattern):
        super().__init__([0], [0], color=color, lw=4, label=label)
        self.hatch_pattern = hatch_pattern

    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        line = super().create_artists(legend, orig_handle,
                                      x0, y0, width, height, fontsize, trans)
        for l in line:
            l.set_hatch(self.hatch_pattern)
        return line

class PLOT(object):
    def __init__(self, X, Y, Descriptions, X_label, Y_label, name, condition=False):
        self.X = X
        self.Y = Y
        self.Desc = Descriptions
        self.XL = X_label
        self.YL = Y_label
        self.name = name
        self.condition = condition
        self.markers = ['H', 'D', 'v', '^', '<', '>', 'd']
        self.Line_style = ['-', ':', '--', '-.', '--']
        # Using a set of visually appealing colors
        self.colors = ['red','green','fuchsia', 'cyan','green','orchid' ,'indigo', 'magenta', 'orange', 'yellow', 'lime', 'gold', 'seagreen']
        self.h = ['','/','.','x']
    def scatter_line(self, Grid, y, Log=False):
        plt.close('all')

        legend_elements = []
        for i in range(len(self.Desc)):
            legend_elements.append(Line2D([0], [0], marker='', color=self.colors[i], lw=3, label=self.Desc[i],
                                          linestyle=self.Line_style[i]))

        fig, ax = plt.subplots()
        ax.legend(handles=legend_elements, fontsize=14, loc='upper left')

        if Log:
            plt.xscale("log")
        if Grid:
            plt.grid(linestyle='--', color='darkblue', linewidth=1.5)

        for j in range(len(self.Y)):
            plt.plot(self.X, self.Y[j], alpha=1, color=self.colors[j], linestyle=self.Line_style[j], linewidth=2.5)
            plt.scatter(self.X, self.Y[j], marker='h', linewidths=0.7, alpha=1, color=self.colors[j])

        plt.ylim(0, y)

        plt.xlabel(self.XL, fontdict={'color': 'darkblue', 'size': 20}, fontsize=17, fontweight='bold')
        plt.ylabel(self.YL, fontdict={'color': 'darkblue', 'size': 20}, fontsize=17, fontweight='bold')
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 2

        ax.spines['top'].set_color('darkblue')
        ax.spines['bottom'].set_color('darkblue')
        ax.spines['left'].set_color('darkblue')
        ax.spines['right'].set_color('darkblue')

        ax.tick_params(axis='x', colors='darkblue', which='both', labelsize=15)
        ax.tick_params(axis='y', colors='darkblue', which='both', labelsize=15)

        plt.xticks(fontsize=15, weight='bold')
        plt.yticks(weight='bold', fontsize=15)
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().xaxis.set_tick_params(width=3)
        plt.gca().yaxis.set_tick_params(width=1.5)
        plt.gca().xaxis.set_tick_params(length=6)
        plt.gca().yaxis.set_tick_params(length=6)

        # Set the frame to be a rectangle with rounded corners
        ax.set_frame_on(True)
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
        plt.tight_layout(rect=[0, 0, 1, 1], pad=0.1)
   #    x_ticks = [0.05, 0.1, 0.15,0.2]

        # Set the x-axis ticks to the specified values
    #   plt.xticks(x_ticks, fontsize=15, weight='bold')
        
        plt.savefig(self.name, format='png', dpi=600)


    def scatter_area(self, Grid, y, Log=False):
        plt.close('all')

        legend_elements = []
        for i in range(len(self.Desc)):
            legend_elements.append(Line2D([0], [0], marker='', color=self.colors[i], lw=5, label=self.Desc[i],
                                          linestyle=self.Line_style[i]))

        fig, ax = plt.subplots()
        ax.legend(handles=legend_elements, fontsize=14, loc='upper right')

        if Log:
            plt.xscale("log")
        if Grid:
            plt.grid(linestyle='--', color='darkblue', linewidth=1.5)

        area_values = []

        for j in range(len(self.Y)):
            plt.plot(self.X, self.Y[j], alpha=1, color=self.colors[j], linestyle=self.Line_style[j], linewidth=2.5)
            plt.scatter(self.X, self.Y[j], marker='h', linewidths=0.7, alpha=1, color=self.colors[j])

            area = np.trapz(self.Y[j], self.X)
            area_values.append(area)

            hatch_pattern = '/' if j == 0 else '\\'
            plt.fill_between(self.X, self.Y[j], alpha=0.5, color=self.colors[j], hatch=hatch_pattern, edgecolor='black')

        plt.ylim(0, y)

        plt.xlabel(self.XL, fontdict={'color': 'darkblue', 'size': 20}, fontsize=17, fontweight='bold')
        plt.ylabel(self.YL, fontdict={'color': 'darkblue', 'size': 20}, fontsize=17, fontweight='bold')
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 2

        ax.spines['top'].set_color('darkblue')
        ax.spines['bottom'].set_color('darkblue')
        ax.spines['left'].set_color('darkblue')
        ax.spines['right'].set_color('darkblue')

        ax.tick_params(axis='x', colors='darkblue', which='both', labelsize=15)
        ax.tick_params(axis='y', colors='darkblue', which='both', labelsize=15)

        plt.xticks(fontsize=15, weight='bold')
        plt.yticks(weight='bold', fontsize=15)
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().xaxis.set_tick_params(width=1.5)
        plt.gca().yaxis.set_tick_params(width=1.5)
        plt.gca().xaxis.set_tick_params(length=6)
        plt.gca().yaxis.set_tick_params(length=6)

        # Set the frame to be a rectangle with rounded corners
        ax.set_frame_on(True)
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
        plt.tight_layout(rect=[0, 0, 1, 1], pad=0.1)

        plt.savefig(self.name, format='png', dpi=600)

        return area_values


    
    
    def Box_Plot(self, Grid, y_range=None):
        color1 = self.colors[0]
        color2 = self.colors[2]
        self.colors[0] = color2
        self.colors[2] = color1
    
        plt.close("all")
        fig, axs = plt.subplots(figsize=(10, 10))  # Adjust the figsize as needed


        plt.rcParams["axes.linewidth"] = 1
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 2
        plt.rcParams['xtick.labelsize'] = 5
        plt.rcParams['ytick.labelsize'] = 5

        font1 = {'family': 'Times Roman', 'color': 'darkblue', 'size': 16}
        fig, axs = plt.subplots()
        axs.set_ylabel(self.YL, fontdict=font1, fontsize='x-large', fontweight='bold')
        axs.set_xlabel(self.XL, fontdict=font1, fontsize='x-large', fontweight='bold')

        flierprops = dict(marker='x', markersize=2)
        medianprops = dict(color="black", linewidth=1.5)
        whiskerprops = dict(linewidth=2)
        capprops = {'linewidth': '2'}
        Elements = []
        if y_range is not None:
            plt.ylim(0,y_range)        
        for j in range(len(self.Y)):
            hatch_pattern = '' if j == 0 else '/' if j == 1 else '\\'  # Customize hatch patterns as needed
        
            # Add a legend entry for each box plot using the custom LegendHandler
            legend_entry = LegendHandler(color=self.colors[j], label=self.Desc[j], hatch_pattern=hatch_pattern)
            Elements.append(legend_entry)
        
            for i in range(len(self.X)):
                position = i + 0.5 * j  # Adjust the multiplier as needed for proper spacing
        
                # Adjust the width parameter to make the box plots wider
                box = axs.boxplot(self.Y[j][i], positions=[position], widths=0.2,  # Adjust the width as needed
                                  notch=False, patch_artist=True,
                                  boxprops=dict(facecolor=self.colors[j], color=self.colors[j],
                                                hatch=hatch_pattern),  # Add hatch pattern here
                                  flierprops=flierprops, medianprops=medianprops, whiskerprops=whiskerprops,
                                  capprops=capprops)
                # Set path effects for rounded corners
                for patch in box['boxes']:
                    patch.set_edgecolor('darkblue')
                    patch.set_linewidth(1.5)
                    patch.set_facecolor(self.colors[j])


        legend_elements = Elements
       # axs.legend(handles=legend_elements, fontsize=12, loc='upper left')
        #plt.legend(handles=Elements, fontsize=12, loc='upper left')
        plt.legend(handles=Elements, fontsize=15, loc='upper left')        
        if Grid:
            plt.grid(linestyle='--', color='darkblue', linewidth=1.5)

        plt.tight_layout()

        # Set the frame to be a rectangle with rounded corners
        for spine in axs.spines.values():
            spine.set_edgecolor('darkblue')
            spine.set_linewidth(1.5)
        
        axs.xaxis.labelpad = 10
        axs.yaxis.labelpad = 10
        axs.set_frame_on(True)
        axs.xaxis.set_tick_params(width=1.5)
        axs.yaxis.set_tick_params(width=1.5)
        axs.xaxis.set_tick_params(length=6)
        axs.yaxis.set_tick_params(length=6)
        axs.set_ylabel(self.YL, fontdict=font1, fontsize='x-large', fontweight='bold', color='darkblue')
        axs.set_xlabel(self.XL, fontdict=font1, fontsize='x-large', fontweight='bold', color='darkblue')
        
        axs.tick_params(axis='x', colors='darkblue', which='both', labelsize=15)
        axs.tick_params(axis='y', colors='darkblue', which='both', labelsize=15)
        
        # Set x-tick positions and labels
        x_positions = [i*1+0.25 for i in range(len(self.X)) ]
        axs.set_xticks(x_positions)
        axs.set_xticklabels([val if j % 2 == 0 else '' for j in range(len(self.Y)) for val in self.X])
        
        plt.tight_layout()
        plt.savefig(self.name, format='png', dpi=600)
        plt.show()














    def Box_Plot_(self, Grid, y_range=None):
        color1 = self.colors[0]
        color2 = self.colors[2]
        self.colors[0] = color2
        self.colors[2] = color1
        y_values_list= self.Y
    
        plt.close("all")
        fig, axs = plt.subplots(figsize=(10, 10))  # Adjust the figsize as needed
    
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 2
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
    
        font1 = {'family': 'Times Roman', 'color': 'darkblue', 'size': 16}
        axs.set_ylabel(self.YL, fontdict=font1, fontsize='x-large', fontweight='bold')
        axs.set_xlabel(self.XL, fontdict=font1, fontsize='x-large', fontweight='bold')
    
        flierprops = dict(marker='x', markersize=2)
        medianprops = dict(color="black", linewidth=1.5)
        whiskerprops = dict(linewidth=1.5)
        capprops = {'linewidth': '1.5'}
        Elements = []
        if y_range is not None:
            plt.ylim(4,y_range)
        counter = 0
        if y_values_list is not None:
            max_len = max(len(y) for y in y_values_list)
            for i in range(max_len):
                positions = [i*2.5 + j * (max_len + 1) * 0.1 for j, y_values in enumerate(y_values_list) if i < len(y_values)]
                
                
                for j, y_values in enumerate(y_values_list):
                    hatch_pattern = self.h[j]  # Customize hatch patterns as needed                    
                                # Add a legend entry for each box plot using the custom LegendHandler
                    legend_entry = LegendHandler(color=self.colors[j], label=self.Desc[j], hatch_pattern=hatch_pattern)
                    if counter<len(y_values_list):
                        Elements.append(legend_entry)
                    counter = counter + 1
                    if i < len(y_values):
        
    
                        # Adjust the width parameter to make the box plots wider
                        box = axs.boxplot(y_values[i], positions=[positions[j]], widths=0.3,  # Adjust the width as needed
                                          notch=False, patch_artist=True,
                                          boxprops=dict(facecolor=self.colors[j], color=self.colors[j],
                                                        hatch=hatch_pattern),  # Add hatch pattern here
                                          flierprops=flierprops, medianprops=medianprops, whiskerprops=whiskerprops,
                                          capprops=capprops)
                        # Set path effects for rounded corners
                        for patch in box['boxes']:
                            patch.set_edgecolor('darkblue')
                            patch.set_linewidth(1.5)
                            patch.set_facecolor(self.colors[j])
    
        legend_elements = Elements
        plt.legend(handles=legend_elements, fontsize=15, loc='upper left')
    
        if Grid:
            plt.grid(linestyle='--', color='darkblue', linewidth=1.5)
    
        plt.tight_layout()
    
        # Set the frame to be a rectangle with rounded corners
        for spine in axs.spines.values():
            spine.set_edgecolor('darkblue')
            spine.set_linewidth(1.5)
    
        axs.xaxis.labelpad = 10
        axs.yaxis.labelpad = 10
        axs.set_frame_on(True)
        axs.xaxis.set_tick_params(width=1.5)
        axs.yaxis.set_tick_params(width=1.5)
        axs.xaxis.set_tick_params(length=6)
        axs.yaxis.set_tick_params(length=6)
        axs.set_ylabel(self.YL, fontdict=font1, fontsize='x-large', fontweight='bold', color='darkblue')
        axs.set_xlabel(self.XL, fontdict=font1, fontsize='x-large', fontweight='bold', color='darkblue')
    
        axs.tick_params(axis='x', colors='darkblue', which='both', labelsize=15)
        axs.tick_params(axis='y', colors='darkblue', which='both', labelsize=15)
    
        # Set x-tick positions and labels
        x_positions = [j*2.5 + 0.6+0.0 * (len(y_values_list[j]) - 1) for j in range(len(y_values_list[0]))]
        axs.set_xticks(x_positions)
        axs.set_xticklabels([val for val in self.X])
    
        plt.tight_layout()
        plt.savefig(self.name, format='png', dpi=600)
        plt.show()




'''


# Example usage
x_values = ['a', 'b', 'c']
Y_values1 = [[1, 2], [4, 5], [7, 8]]
Y_values2 = [[10, 11], [13, 14], [16, 17]]
Y_values3 = [[19, 20], [22, 23], [25, 26]]
Y_values4 = [[28, 29], [31, 32], [34, 35]]

Y = [Y_values1, Y_values2, Y_values3, Y_values4]
# Plotting Box Plot
box_plot_instance = PLOT(
    X=x_values,
    Y= Y,  # Pass None for Y as it will be provided through y_values_list
    Descriptions=['Box1', 'Box2', 'Box3', 'Box4'],
    X_label='X-axis',
    Y_label='Y-axis',
    name='box_plot.png',
    condition=False
)

box_plot_instance.Box_Plot_(Grid=True, y_range=(0, 40))


'''





       
'''        
    '/': Diagonal lines from top-left to bottom-right
    '\': Diagonal lines from top-right to bottom-left
    '-': Horizontal lines
    '|': Vertical lines
    '+': Crosshatch
    'x': X pattern
    'o': Circles
    'O': Larger circles
    '.': Dots
# Example usage
x_values = ['a', 'b', 'c']
y_values1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
y_values2 = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]

# Plotting Box Plot
box_plot_instance = PLOT(
    X=x_values,
    Y=[y_values1, y_values2],
    Descriptions=['Box1', 'Box2'],
    X_label='X-axis',
    Y_label='Y-axis',
    name='box_plot.png',
    condition=False
)

box_plot_instance.Box_Plot(Grid=True,y_range=(0, 10))
    '''

'''
# Example usage
x_values = np.linspace(0, 10, 100)
y_values1 = np.sin(x_values)
y_values2 = np.cos(x_values)

# Plotting
plot_instance = PLOT(
    X=x_values,
    Y=[y_values1, y_values2],
    Descriptions=['Sin', 'Cos'],
    X_label='X-axis',
    Y_label='Y-axis',
    name='example_plot.png',
    condition=False
)

# Scatter Area Plot
areas = plot_instance.scatter_area(Grid=True, y=1.5, Log=False)

# Output areas
for i, area in enumerate(areas):
    print(f"Area below {plot_instance.Desc[i]}: {area}")
'''