
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

    # FIXME find a nice way to order the dict for consistent plotting!

    if sort_keys:
        # order the dictionary so the x-axis is in the correct order
        # d = {int(k): v for k, v in input_dict.items()}
        d = {(sum(v)/len(v)): v for k, v in input_dict.items()}
        ordered_dict = collections.OrderedDict(sorted(d.items()))
    else:
        ordered_dict = input_dict

    ordered_dict = {'Random': [787.0374000000003, 660.5401999999996, 674.9678, 773.5338000000003, 774.5124000000001, 754.6317999999998, 667.2614000000004, 801.1924000000006, 666.6588000000002, 764.3524000000003, 655.8844000000004, 698.9474000000002, 710.0844000000003, 849.0284000000001, 849.2644000000003], 
'Heuristic_with_noise': [822.8373000000004, 673.3290999999999, 779.2187000000001, 647.2036999999998, 832.4403000000001, 718.6366999999999, 807.4912999999998, 730.8203000000002, 639.2497000000001, 769.7703000000001, 817.7093, 799.9893000000003, 773.3903, 836.8523000000001, 805.6343000000005], 
'Fixed_wing_PPO': [828.2233000000001, 722.1570999999998, 769.8326999999999, 754.4066999999999, 849.3033000000004, 790.5817000000001, 830.4573000000003, 854.9223000000002, 790.4897, 861.2443000000001, 807.0943, 829.8323000000004, 838.2872999999998, 894.3803000000004, 854.0373000000004], 
'Rotary_wing_PPO': [866.9923000000002, 747.3090999999997, 803.4096999999999, 794.4397000000005, 876.8863000000001, 793.3976999999998, 837.7352999999999, 843.1982999999997, 817.3607000000006, 862.9243, 832.0573000000003, 863.4153000000005, 862.2973, 889.8873000000001, 856.0113000000002],
'Heuristic_no_noise': [886.0033, 772.0600999999998, 841.9597000000001, 810.7157000000001, 882.9393, 805.6547000000002, 864.4073000000003, 861.2723000000001, 837.6527000000006, 887.8143000000001, 862.4783000000001, 878.5543000000006, 873.1832999999999, 914.6333, 866.0593000000005], 
'Relaxation': [925.3533000000003, 809.3981000000003, 880.3726999999999, 842.6587000000006, 918.9993000000002, 842.2406999999997, 900.6833000000005, 897.6393, 877.2047000000003, 934.5893000000005, 900.9653000000006, 918.4753000000002, 911.7072999999997, 954.2323000000006, 912.5443]}

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

# Mean: 0.5101359009742736
# std: 0.011494420189369551

    rewards_dict: dict = {
        '1D CNN': (0.6843001842498779, 0.6834026575088501, 0.6951318979263306, 0.6628499031066895, 0.6842191219329834),
        'Conv-LSTM': (0.6617647409439087, 0.6507555246353149, 0.6550811529159546, 0.6752890348434448, 0.6520588397979736),
        'XGBoost': (0.6055426,  0.63983773, 0.63319093, 0.58934513, 0.58217766),
        'Logistic Regression': (0.62275862, 0.6030426,  0.59708499, 0.57301101, 0.58218274),
        'Random Forest': (0.54240872, 0.54993408, 0.57404634, 0.53398502, 0.56172589),
        'LSTM': (0.5106490850448608, 0.5, 0.5086206793785095, 0.5, 0.531409740447998)
    }

    boxplot_dict(fname=f'model_results', input_dict=rewards_dict, 
                boxplot_title='AUC Scores For Each Model - 5 Fold Cross Validation', 
                x_label='Model Being Evaluated', 
                y_label='Area Under The ROC Curve', image_type='png',
                save_results=False, show_results=True, sort_keys=True, 
                path_to_figs=figure_dir)
