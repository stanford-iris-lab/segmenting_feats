import matplotlib.pyplot as plt
import numpy as np
import pandas

hex_colors = [
    "#FF6150",
    "#134E6F",
    "#1AC0C6",
    "#FFA822",
    "#DEE0E6",
    "#091A29"
]

lighter_hex_colors = [
    "#ff8f82",
    "#4d83a3",
    "#92d4d6",
    "#FFA822",
    "#DEE0E6",
    "#091A29"
]

noft = pandas.read_csv('./run_data/mvp_dino_r3m_noft.csv')
ft = pandas.read_csv('./run_data/dino_r3m_ft.csv')

lighting_columns = ['eval_successbrighter', 'eval_successdarker', 'eval_successleft', 'eval_successright']
texture_columns = ['eval_successmetal2', 'eval_successtile1', 'eval_successwood2']
distractor_columns = ['eval_successhard']
all_dist_shift_columns = lighting_columns + texture_columns + distractor_columns

# add lighting, texture, distractor, and all average test columns
noft['Train Dist.'] = noft['eval_success']
noft['Test (Lighting)'] = noft[lighting_columns].mean(axis=1)
noft['Test (Texture)'] = noft[texture_columns].mean(axis=1)
noft['Test (Distractors)'] = noft[distractor_columns].mean(axis=1)
noft['Test (Avg)'] = noft[all_dist_shift_columns].mean(axis=1)
ft['Train Dist.'] = ft['eval_success']
ft['Test (Lighting)'] = ft[lighting_columns].mean(axis=1)
ft['Test (Texture)'] = ft[texture_columns].mean(axis=1)
ft['Test (Distractors)'] = ft[distractor_columns].mean(axis=1)
ft['Test (Avg)'] = ft[all_dist_shift_columns].mean(axis=1)

# aggregate results by seed
r3m_noft = noft[noft['embedding'] == 'resnet50'].groupby('seed').mean()
r3m_ft = ft[ft['embedding'] == 'resnet50'].groupby('seed').mean()
dino_noft = noft[noft['embedding'] == 'dino'].groupby('seed').mean()
dino_ft = ft[ft['embedding'] == 'dino'].groupby('seed').mean()
# plot standard error by seed
r3m_noft_mean = r3m_noft.mean()
r3m_noft_err = (r3m_noft.std()*1.96) / np.sqrt(3)
r3m_ft_mean = r3m_ft.mean()
r3m_ft_err = (r3m_ft.std()*1.96) / np.sqrt(3)
dino_noft_mean = dino_noft.mean()
dino_noft_err = (dino_noft.std()*1.96) / np.sqrt(3)
dino_ft_mean = dino_ft.mean()
dino_ft_err = (dino_ft.std()*1.96) / np.sqrt(3)

# make bar chart (mvp vs dino vs r3m figure)
# labels = ['MVP', 'MVP (FT)', 'R3M', 'R3M (FT)', 'DiNo', 'DiNo (FT)']
labels = ['Train Dist.', 'Test (Lighting)', 'Test (Texture)', 'Test (Distractors)', 'Test (Avg)']
r3m_noft_means = [r3m_noft_mean[label] for label in labels]
r3m_ft_means = [r3m_ft_mean[label] for label in labels]
dino_noft_means = [dino_noft_mean[label] for label in labels]
dino_ft_means = [dino_ft_mean[label] for label in labels]
r3m_noft_errs = [r3m_noft_err[label] for label in labels]
r3m_ft_errs = [r3m_ft_err[label] for label in labels]
dino_noft_errs = [dino_noft_err[label] for label in labels]
dino_ft_errs = [dino_ft_err[label] for label in labels]

# ft and noft
def plot_ft_and_noft():
    x = np.arange(len(labels))*2.5  # the label locations
    width = 0.35  # the width of the bars
    delta = .01

    fig, ax = plt.subplots(figsize=(20,6))
    rects1 = ax.bar(x - width*2.5 - delta, r3m_noft_means, width, yerr=r3m_noft_errs,  label='MVP', color=lighter_hex_colors[2])
    rects2 = ax.bar(x - width*1.5 - delta, r3m_noft_means, width, yerr=r3m_noft_errs,  label='R3M', color=lighter_hex_colors[1])
    rects3 = ax.bar(x - width/2 - delta, dino_noft_means, width, yerr=dino_noft_errs,  label='DiNo', color=lighter_hex_colors[0])
    rects4 = ax.bar(x + width/2 + delta, r3m_ft_means, width, yerr=r3m_ft_errs, label='MVP (FT)', color=hex_colors[2])
    rects5 = ax.bar(x + width*1.5 + delta, r3m_ft_means, width, yerr=r3m_ft_errs, label='R3M (FT)', color=hex_colors[1])
    rects6 = ax.bar(x + width*2.5 + delta, dino_ft_means, width, yerr=dino_ft_errs, label='DiNo (FT)', color=hex_colors[0])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Success', fontsize=18)
    # ax.set_title('Zero-Shot Transfer Performance', fontsize=18)
    ax.set_xticks(x, labels, fontsize=18)
    ax.set_ylim(0,22)
    plt.axvline(x=1.25, color='black', linestyle='--')
    ax.legend(fontsize=18, ncol=2)
    ax.spines[['right', 'top']].set_visible(False)

    # ax.bar_label(rects4, padding=3)
    # ax.bar_label(rects5, padding=3)
    # ax.bar_label(rects6, padding=3)

    fig.tight_layout()
    plt.savefig('mvp_dino_r3m_ft_and_noft.png')
plot_ft_and_noft()

# just noft
def plot_just_noft():
    x = np.arange(len(labels))*1.2  # the label locations
    width = 0.3  # the width of the bars
    delta = .01

    fig, ax = plt.subplots(figsize=(12,6))
    rects1 = ax.bar(x - width, r3m_noft_means, width, yerr=r3m_noft_errs,  label='MVP', color=hex_colors[2])
    rects2 = ax.bar(x, r3m_noft_means, width, yerr=r3m_noft_errs,  label='R3M', color=hex_colors[1])
    rects3 = ax.bar(x + width, dino_noft_means, width, yerr=dino_noft_errs,  label='DiNo', color=hex_colors[0])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Success')
    ax.set_title('Zero-Shot Transfer Performance')
    ax.set_xticks(x, labels)
    plt.axvline(x=.6, color='black', linestyle='--')
    ax.legend()
    ax.spines[['right', 'top']].set_visible(False)

    # ax.bar_label(rects4, padding=3)
    # ax.bar_label(rects5, padding=3)
    # ax.bar_label(rects6, padding=3)

    fig.tight_layout()
    plt.savefig('mvp_dino_r3m_noft.png')
