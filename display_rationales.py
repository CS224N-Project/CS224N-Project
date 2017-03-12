import sys
import gzip
import copy
import scipy.stats


# input_path = sys.argv[0]
# output_path = sys.argv[1]
input_path = '/Users/stanford/Desktop/Winter2017/CS224n/FinalProject/beer/reviews.aspect1.small.heldout.txt.gz'
output_path = '/Users/stanford/Desktop/Winter2017/CS224n/FinalProject/beer/reviews.aspect1.highlighted.heldout.txt.gz'

line_list = [ ]
with gzip.open(input_path) as input_file:
    for line in input_file:
        y, sep, x = line.partition("\t")
        line_list.append(x.split())

p = 0.2
mask_list = [ ]
for line in line_list:
	mask_list.append(scipy.stats.bernoulli.rvs(p, size=len(line)))

highlighted_list = copy.deepcopy(line_list)
for line_idx, line in enumerate(line_list):
	for word_idx, word in enumerate(line):
	    if mask_list[line_idx][word_idx]:
	    	highlighted_list[line_idx][word_idx] = "<<" + highlighted_list[line_idx][word_idx].upper() + ">>"

with gzip.open(output_path, "w") as fout:
     for line in highlighted_list:
         fout.write(' '.join(line) + '\n\n')

