import argparse
import csv
import itertools

from flair.embeddings import BertEmbeddings
from flair.data import Sentence

import progressbar

import numpy as np
import pandas


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('tsv_path', help='Path to TSV')
    return parser.parse_args()


def main():
    args = _parse_args()
    tsv_path = args.tsv_path

    embedding = BertEmbeddings('bert-base-cased')

    sentences = [[]]
    with open(tsv_path, 'r') as f:
        for i, l in enumerate(f.readlines()):
            if l.strip():
                token, *_ = l.strip().split('\t')
                sentences[-1].append(token.lower())
            else:
                sentences.append([])

    f_sentences = [Sentence(' '.join(s)) for s in sentences]

    for s in progressbar.progressbar(f_sentences):
        embedding.embed(s)

        for t in s:
            print('\t'.join(t.embedding.numpy().astype(str)))
        print()

        s.clear_embeddings()


if __name__ == '__main__':
    main()
