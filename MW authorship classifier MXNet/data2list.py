import os
import random
import argparse
import json
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../python"))

def list_file(root, recursive, prefix):
    """Traverses the root of directory that contains the samples and
    generates a list iterator.
    Parameters
    ----------
    root: string
    recursive: bool
    Returns
    -------
    image iterator that contains all files under the specified path
    """
    i = 0
    label_to_class ={}
    if recursive:
        cat = {}
        for path, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files.sort()
            for fname in files:
                fpath = os.path.join(path, fname)
                if os.path.isfile(fpath) :
                    if path not in cat:
                        cat[path] = len(cat)
                    yield (i, os.path.relpath(fpath, root), cat[path])
                    i += 1
        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            print(os.path.relpath(k, root), v)
            label_to_class[v] = os.path.relpath(k, root)
    else:
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            if os.path.isfile(fpath):
                yield (i, os.path.relpath(fpath, root), 0)
                i += 1
    with open(prefix  + '_label_to_name.json', 'w') as fp:
        json.dump(label_to_class, fp)

def write_list(path_out, file_list):
    """Helper function to write the meta-data information
    The format is as below,
    integer_file_index \t float_label_index \t path_to_file
    Note that the blank between number and tab is only used for readability.
    Parameters
    ----------
    path_out: string
    file_list: list
    """
    with open(path_out, 'w') as fout:
        for i, item in enumerate(file_list):
            line = '%d\t' % item[0]
            for j in item[2:]:
                line += '%f\t' % j
            line += '%s\n' % item[1]
            fout.write(line)

def make_list(args):
    """Generates .lst file.
    Parameters
    ----------
    args: object that contains all the arguments
    """
    file_list = list_file(args.root, args.recursive, args.prefix)
    file_list = list(file_list)
    if args.shuffle is True:
        random.seed(100)
        random.shuffle(file_list)
    N = len(file_list)
    
    sep = int(N * args.train_ratio)
    sep_test = int(N* args.test_ratio)
    if args.train_ratio == 1.0:
        write_list(args.prefix + '.lst', file_list)
    else:
        if args.test_ratio:
            write_list(args.prefix  + '_test.lst', file_list[:sep_test])
        if args.train_ratio + args.test_ratio < 1.0:
            write_list(args.prefix + '_val.lst', file_list[sep_test + sep:])
        write_list(args.prefix  + '_train.lst', file_list[sep_test:sep_test + sep])

def parse_args():
    """Defines all arguments.
    Returns
    -------
    args object that contains all the params
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create a file list by reading from a file list')
    parser.add_argument('prefix', help='prefix of output lst files.')
    parser.add_argument('root', help='path to folder containing files.')

    cgroup = parser.add_argument_group('Options for creating data lists')
    
    cgroup.add_argument('--train-ratio', type=float, default=1.0,
                        help='Ratio of files to be used for training.')
    cgroup.add_argument('--test-ratio', type=float, default=0,
                        help='Ratio of files to use for testing.')
    cgroup.add_argument('--recursive', action='store_true',
                        help='If true recursively walk through subdirs and assign a unique label\
        to the files in each folder. Otherwise only include files in the root folder\
        and give them label 0.')
    cgroup.add_argument('--no-shuffle', dest='shuffle', action='store_false',
                        help='If this is passed, \
        data2list will not randomize the file order in <prefix>.lst')
    
    args = parser.parse_args()
    args.prefix = os.path.abspath(args.prefix)
    args.root = os.path.abspath(args.root)
    return args

if __name__ == '__main__':
    args = parse_args()

    make_list(args)
   
