import os
import sys
from tqdm import tqdm
from time import time
import numpy as np
from sklearn.metrics import accuracy_score

tags = ['javascript', 'java', 'python', 'ruby', 'php', 'c++', 'c#', 'go', 'scala', 'swift']

def main(fin, fout):
    
    n_corrupted_lines = 0
    n_written_lines = 0
    
    with tqdm(total=9999994) as pbar:
        with open(fin, 'r') as reader:
            with open(fout, 'bw') as writer:
                while True:
                    line = reader.readline()
                    pbar.update(1)                    
                    if len(line) == 0:
                        break
                    text_labels = line[:-1].split("\t")
                    if len(text_labels) != 2:
                        n_corrupted_lines += 1
                        continue
                    text, labels = text_labels
                    labels = labels.split(' ')
                    found_tags = [l for l in labels if l in tags]
                    if len(found_tags) != 1:
                        continue
                    text = text.strip().replace(':', ' ').replace('|',' ')
                    label = tags.index(found_tags[0]) + 1
                    writer.write("{} | {}\n".format(label, text).encode('utf-8'))
                    n_written_lines += 1
    print("{} lines selected, {} lines corrupted.".format(n_written_lines, n_corrupted_lines))


if __name__ == '__main__':
    fin = sys.argv[1]
    fout = sys.argv[2]
    assert os.path.exists(fin)
    if fin is None or fout is None:
        print('Incorrect args:\nfin - {}\nfout - {}'.format(fin, fout))
        sys.exit(-1)
    main(fin, fout)
    print('Done')