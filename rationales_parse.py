import json
import gzip

def myio_read_rationales(path):
    data = [ ]
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            item = json.loads(line)
            data.append(item)
    return data

rationale_data = myio_read_rationales('/Users/stanford/Desktop/Winter2017/CS224n/FinalProject/beer/annotations.json')

with gzip.open('/Users/stanford/Desktop/Winter2017/CS224n/FinalProject/beer/annotations.txt.gz', "w") as fout:
	for x in rationale_data:
		scores_str = ' '.join(['{:.2f}'.format(num) for num in x["y"]])
 		review_str = ' '.join([word.encode('ascii','ignore') for word in x["x"]])
 		fout.write(scores_str + '\t' + review_str + '\n')

