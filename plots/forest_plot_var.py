import forestplot as fp
import pandas as pd
import matplotlib.pyplot as plt

# 5-shot VOC1
# Note: performance decrease on 5-shot VOC1, increase on 10-shot COCO when using split-merged dataloader with same results as CFA using both CFA and original

data = [
    {'name': 'DeFRCN (Published, Averaged)', 'nAP': 37.3, 'ci': 0.8},
    {'name': 'DeFRCN (Reproduced, Averaged)', 'nAP': 38.1, 'ci': 1.1},
    {'name': 'DeFRCN (Diff. RNG, Averaged)', 'nAP': 38.1, 'ci': 0.9},
    {'name': 'DeFRCN (Diff. surgery, Averaged)', 'nAP': 38.1, 'ci': 0.9},
    # {'name': 'DeFRCN (Diff. dataloader, Averaged)', 'nAP': 37.4, 'ci': 0.8},
    {'name': 'DeFRCN (Published, FSRW)', 'nAP': 40.8, 'ci': 0.},
    {'name': 'DeFRCN (Reproduced, FSRW)', 'nAP': 41.4, 'ci': 0.},
    {'name': 'DeFRCN (Diff. RNG, FSRW)', 'nAP': 40.8, 'ci': 0.},
    {'name': 'DeFRCN (Diff. surgery, FSRW)', 'nAP': 41.1, 'ci': 0.},
    # {'name': 'DeFRCN (Diff. dataloader, FSRW)', 'nAP': 39.9, 'ci': 0.},
]
for d in data:
    d['cil'] = d['nAP'] - d['ci']
    d['cih'] = d['nAP'] + d['ci']

df = pd.DataFrame(data)

plot = fp.forestplot(dataframe=df, estimate='nAP', model_col='nAP50', ll='cil', hl='cih', varlabel='name',
                     annote=[], figsize=(10, 4),
                     xticks=[x / 10 for x in range(360, 430, 10)], xlabel='nAP on VOC-Split 1 (95% CI)',
                     color_alt_rows=True,
                     **{"xline": 32.4, "xlinestyle": (1, (10, 5)),
                        "xlinecolor": "#808080",  # gray color for x-reference line
                        "xtick_size": 12,  # adjust x-ticker fontsize
                        }
                     )

# fig = plt.figure()
# fig.add_axes(plot)
# plt.show() # bbox_inches='tight'
plt.savefig('randomness_voc_forestplot.png', dpi="figure", bbox_inches="tight")
# plt.show()
