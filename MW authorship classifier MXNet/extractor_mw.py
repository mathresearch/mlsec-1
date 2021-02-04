import re
import numpy as np
import mmh3

chars = r" -~"
min_length = 5
string_regexp = '[%s]{%d,}' % (chars, min_length)
"""
The %s and the %d will be replaced with the values of the chars and min_length variables. 
The resulting string will be "[ -~]{5,}". 
[ -~] means "A character within the range of space and tilde". 
{5,} means "five or more of the preceding value". 
All together, the pattern means "five or more printable ascii characters".
"""

def extract_features(string, hash_dim, split_regex= string_regexp):
        tokens = re.split(pattern=split_regex, string=string)
        hash_buckets = [(mmh3.hash(w) % hash_dim) for w in tokens]
        buckets, counts = np.unique(hash_buckets, return_counts=True)
        feature_values = np.zeros(hash_dim)
        for bucket, count in zip(buckets, counts):
            feature_values[bucket] = count
        return feature_values