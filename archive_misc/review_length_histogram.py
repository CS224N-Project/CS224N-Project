
import gzip
import numpy as np
import math
from matplotlib import pyplot as plt

input_path = '/Users/stanford/Desktop/Winter2017/CS224n/FinalProject/beer/reviews.aspect1.train.txt.gz'

line_list = [ ]
with gzip.open(input_path) as input_file:
    for line in input_file:
        y, sep, x = line.partition("\t")
        line_list.append(x.split())

line_lengths = [ ]
for line in line_list:
    line_lengths.append(len(line))


data = line_lengths

bins = np.linspace(math.ceil(min(data)), 
                   math.floor(max(data)),
                   20) # fixed number of bins

plt.xlim([min(data)-5, max(data)+5])

plt.hist(data, bins=bins, alpha=0.5)
plt.title('Review lengths data (fixed number of bins)')
plt.xlabel('variable X (20 evenly spaced bins)')
plt.ylabel('count')

plt.show()