from __future__ import print_function
import os
import sys
import random
import argparse
import time
import re
import mmh3
import pickle
import zipfile

import numpy as np
import mxnet as mx

from extractor_mw import extract_features

try:
    import Queue as queue
except ImportError:
    import queue

try:
    import multiprocessing
except ImportError:
    multiprocessing = None



def read_list(path_in):
    """Reads the .lst file and generates corresponding iterator.
    Parameters
    ----------
    path_in: string
    Returns
    -------
    item iterator that contains information in .lst file
    """
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            line_len = len(line)
            # check the data format of .lst file
            if line_len < 3:
                print('lst should have at least has three parts, but only has %s parts for %s' % (line_len, line))
                continue
            try:
                item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            except Exception as e:
                print('Parsing lst met error for %s, detail: %s' % (line, e))
                continue
            yield item


def data_encode(args, idx, item, q_out):
    """Reads, preprocesses, packs the data of feature size and puts it back into the output queue.
    Parameters
    ----------
    args: object
    idx: int
    item: list
    q_out: queue
    """
    filepath = os.path.join(args.root, item[1])
    header = mx.recordio.IRHeader(0, item[2], item[0], 0)

    
    
    pswd = b'infected'

    with zipfile.ZipFile(filepath) as fn:
        fn.extractall(pwd =pswd)
        tmp=fn.namelist()[0]
    with open(tmp, encoding="latin-1") as f:
        content = f.read()
    if os.path.exists(tmp):
       os.remove(tmp)
    else:
        print("The file does not exist")
    data = extract_features(content, hash_dim=args.feature_size)
    data = pickle.dumps(data)
    s = mx.recordio.pack(header, data)
    q_out.put((idx, s, item))        

def read_worker(args, q_in, q_out):
    """Function that will be spawned to fetch data
    from the input queue and put it back to output queue.
    Parameters
    ----------
    args: object
    q_in: queue
    q_out: queue
    """
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        data_encode(args, i, item, q_out)

def write_worker(q_out, fname, working_dir):
    """Function that will be spawned to fetch processed data
    from the output queue and write to the .rec file.
    Parameters
    ----------
    q_out: queue
    fname: string
    working_dir: string
    """
    pre_time = time.time()
    count = 0
    fname = os.path.basename(fname)
    fname_rec = os.path.splitext(fname)[0] + '.rec'
    fname_idx = os.path.splitext(fname)[0] + '.idx'
    record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                           os.path.join(working_dir, fname_rec), 'w')
    buf = {}
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            i, s, item = deq
            buf[i] = (s, item)
        else:
            more = False
        while count in buf:
            s, item = buf[count]
            del buf[count]
            if s is not None:
                record.write_idx(item[0], s)

            if count % 1000 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', count)
                pre_time = cur_time
            count += 1

def parse_args():
    """Defines all arguments.
    Returns
    -------
    args object that contains all the params
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create a record database by reading from a data list')
    parser.add_argument('prefix', help='prefix of input/output lst and rec files.')
    parser.add_argument('root', help='path to folder containing images.')

    
    rgroup = parser.add_argument_group('Options for creating database')

    rgroup.add_argument('--num-thread', type=int, default=1,
                        help='number of thread to use for encoding. order of images will be different\
        from the input list if >1. the input list will be modified to match the\
        resulting order.')
    rgroup.add_argument('--pack-label', action='store_true',
        help='Whether to also pack multi dimensional label in the record file')
    parser.add_argument('--feature-size', type=int, default=1024,
                        help="Feature encoding size of a data")
    args = parser.parse_args()
    args.prefix = os.path.abspath(args.prefix)
    args.root = os.path.abspath(args.root)
    return args

if __name__ == '__main__':
    args = parse_args()
   

    if os.path.isdir(args.prefix):
        working_dir = args.prefix
    else:
        working_dir = os.path.dirname(args.prefix)
    files = [os.path.join(working_dir, fname) for fname in os.listdir(working_dir)
                if os.path.isfile(os.path.join(working_dir, fname))]
    count = 0
    for fname in files:
            if fname.startswith(args.prefix) and fname.endswith('.lst'):
                print('Creating .rec file from', fname, 'in', working_dir)
                count += 1
                data_list = read_list(fname)
                # -- write_record -- #
                if args.num_thread > 1 and multiprocessing is not None:
                    q_in = [multiprocessing.Queue(1024) for i in range(args.num_thread)]
                    q_out = multiprocessing.Queue(1024)
                    # define the process
                    read_process = [multiprocessing.Process(target=read_worker, args=(args, q_in[i], q_out)) \
                                    for i in range(args.num_thread)]
                    # process data with num-thread process
                    for p in read_process:
                        p.start()
                    # only use one process to write .rec to avoid race-condtion
                    write_process = multiprocessing.Process(target=write_worker, args=(q_out, fname, working_dir))
                    write_process.start()
                    # put the image list into input queue
                    for i, item in enumerate(data_list):
                        q_in[i % len(q_in)].put((i, item))
                    for q in q_in:
                        q.put(None)
                    for p in read_process:
                        p.join()

                    q_out.put(None)
                    write_process.join()
                else:
                    print('multiprocessing not available, fall back to single threaded encoding')
                    try:
                        import Queue as queue
                    except ImportError:
                        import queue
                    q_out = queue.Queue()
                    fname = os.path.basename(fname)
                    fname_rec = os.path.splitext(fname)[0] + '.rec'
                    fname_idx = os.path.splitext(fname)[0] + '.idx'
                    record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                                           os.path.join(working_dir, fname_rec), 'w')
                    cnt = 0
                    pre_time = time.time()
                    for i, item in enumerate(data_list):
                        data_encode(args, i, item, q_out)
                        if q_out.empty():
                            continue
                        _, s, _ = q_out.get()
                        record.write_idx(item[0], s)
                        if cnt % 1000 == 0:
                            cur_time = time.time()
                            print('time:', cur_time - pre_time, ' count:', cnt)
                            pre_time = cur_time
                        cnt += 1
    if not count:
        print('Did not find and list file with prefix %s'%args.prefix)
