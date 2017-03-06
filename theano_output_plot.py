
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

file_name = 'theano_output_2017-03-04'
output_string = open(file_name + '.txt').read()

output_data = []
output_data.append([int(find_str[16:-3]) for find_str in re.compile('Generator Epoch \d+.00').findall(output_string)])
output_data.append([float(find_str[6:]) for find_str in re.compile('costg=\d+.\d+').findall(output_string)])
output_data.append([float(find_str[6:]) for find_str in re.compile('scost=\d+.\d+').findall(output_string)])
output_data.append([float(find_str[6:]) for find_str in re.compile('lossg=\d+.\d+').findall(output_string)])
output_data.append([float(find_str[5:]) for find_str in re.compile('p\[1\]=\d+.\d+').findall(output_string)])
output_data.append([float(find_str[4:]) for find_str in re.compile('\|g\|=\d+.\d+').findall(output_string)])
output_data.append([float(find_str[-6:]) for find_str in re.compile('\|g\|=\d+.\d+ \d+.\d+').findall(output_string)])
output_data.append([float(find_str[1:-1]) for find_str in re.compile('\[\d+.\d+m').findall(output_string)])
output_data.append([float(find_str[14:]) for find_str in re.compile('sampling devg=\d+.\d+').findall(output_string)])
output_data.append([float(find_str[5:]) for find_str in re.compile('mseg=\d+.\d+').findall(output_string)])
output_data.append([float(find_str[10:]) for find_str in re.compile('avg_diffg=\d+.\d+').findall(output_string)])
output_data.append([float(find_str[6:]) for find_str in re.compile('p\[1\]g=\d+.\d+').findall(output_string)])
output_data.append([float(find_str[9:]) for find_str in re.compile('best_dev=\d+.\d+').findall(output_string)])
output_data.append([float(find_str[15:]) for find_str in re.compile('rationale mser=\d+.\d+').findall(output_string)])
output_data.append([float(find_str[6:]) for find_str in re.compile('p\[1\]r=\d+.\d+').findall(output_string)])
output_data.append([float(find_str[6:]) for find_str in re.compile('prec1=\d+.\d+').findall(output_string)])
output_data.append([float(find_str[6:]) for find_str in re.compile('prec2=\d+.\d+').findall(output_string)])

output_data_df = pd.DataFrame(output_data[0])
for i in range(1, len(output_data)):
	output_data_df = pd.concat([output_data_df, pd.DataFrame(output_data[i])], axis=1)

output_data_df.columns = ['epoch', 'costg', 'scost', 'lossg', 'p1', 'g_1', 'g_2', 'epoch_time', \
'sampling_devg', 'mseg', 'avg_diffg', 'p1_g', 'best_dev', 'rationale_mser', 'p1_r', 'prec1', 'prec2']

output_data_df.to_csv(file_name + '.csv')

fig, ax = plt.subplots()
labels = []
for idx, val in enumerate(['costg', 'mseg']):
	ax = output_data_df.plot(ax=ax, kind='line', x='epoch', y=val)
	labels.append(val)
lines, _ = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
fig = ax.get_figure()
fig.savefig(file_name + '_1.png')

fig = output_data_df.plot(x='epoch', y='prec1').get_figure()
fig.savefig(file_name + '_2.png')

