import os
import numpy as np
import pandas as pd
import pickle

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]
    dirs = os.listdir(dpath)
    for idx, e in enumerate(summary_iterators):
        tags = e.Tags()['scalars']
        out = defaultdict(list) 
        pickle_file = f'{os.path.join(dpath, dirs[idx])}.p'

        for tag in tags:
            out[tag] = [k.value for k in e.Scalars(tag)]
        
        with open(pickle_file, 'wb') as fp:
            pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    path = "/home/emil//Documents/Temporal-SBMC-extension/output/emil/trained_models/final_v3/csv_data"
    tabulate_events(path)