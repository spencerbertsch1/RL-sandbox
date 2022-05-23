
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import collections

def boxplot_dict(fname: str, input_dict: dict, boxplot_title: str, x_label: str, y_label: str, path_to_figs: str, 
                 image_type: str = 'svg', save_results: bool = False, show_results: bool = True, sort_keys: bool = False):
    """
    Author: Spencer Bertsch
    Date: November 2021

    This is a nice general purpose plotting function that I find myself reusing a lot. 

    Generate a boxplot given a dictionary of keys to lists or tuples of ints or floats. The resulting image can be
    saved using different file formats such as .png or .jpeg, but .svg provides the best clarity in papers and
    presentations.
    input_dict = {
        'key1': (2.3, 3.5, 2.7, 2.8, ...),
        'key2': (6.1, 6.3, 5.8, 6.7, ...),
        ...
    }
    :param input_dict:
    :return: NA - saves and displays an image of the resulting figure, doesn't return anything.
    """

    if sort_keys:
        # order the dictionary so the x-axis is in the correct order
        d = {int(k): v for k, v in input_dict.items()}
        ordered_dict = collections.OrderedDict(sorted(d.items()))
    else:
        ordered_dict = input_dict

    # here we need to turn our dict into 2 lists for plotting
    key_list = []
    value_list = []
    for key, val in ordered_dict.items():
        key_list.append(key)
        value_list.append(tuple(val))

    x_tick_list = [x+1 for x in range(len(key_list))]
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f' \n {boxplot_title}', fontsize=16)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(f'{y_label} \n', fontsize=16)
    plt.boxplot(value_list, showmeans=True)  # <-- use patch_artist=True to alter colors
    plt.xticks(x_tick_list, key_list)

    # add custom grid to the plot
    plt.grid(which='major', axis='y', color='lightgrey', linestyle='--', linewidth=0.5)

    if save_results:
        fig_name: str = f'{fname}.{image_type}'
        full_path: Path = f'{path_to_figs}{fig_name}'
        plt.savefig(str(full_path))
        print(f'Figure saved successfully.')

    if show_results:
        plt.show()
