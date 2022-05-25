
from pathlib import Path
import matplotlib.pyplot as plt
# plt.style.use("fivethirtyeight")
plt.style.use("tableau-colorblind10")
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
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(f' \n {boxplot_title}', fontsize=22)
    plt.xlabel(x_label, fontsize=19)
    plt.ylabel(f'{y_label} \n', fontsize=19)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
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


# some test code
if __name__ == "__main__":

    figure_dir = '/'

    rewards_dict: dict = {
        '1D CNN': (0.6843001842498779, 0.6834026575088501, 0.6951318979263306, 0.6628499031066895, 0.6842191219329834),
        'Conv-LSTM': (0.6617647409439087, 0.6507555246353149, 0.6550811529159546, 0.6752890348434448, 0.6520588397979736),
        'XGBoost': (0.6055426,  0.63983773, 0.63319093, 0.58934513, 0.58217766),
        'Logistic Regression': (0.62275862, 0.6030426,  0.59708499, 0.57301101, 0.58218274),
        'Random Forest': (0.54240872, 0.54993408, 0.57404634, 0.53398502, 0.56172589)
    }

    boxplot_dict(fname=f'model_results', input_dict=rewards_dict, 
                boxplot_title='AUC Scores For Each Model - 5 Fold Cross Validation', 
                x_label='Model Being Evaluated', 
                y_label='Area Under The ROC Curve', image_type='png',
                save_results=False, show_results=True, sort_keys=False, 
                path_to_figs=figure_dir)
