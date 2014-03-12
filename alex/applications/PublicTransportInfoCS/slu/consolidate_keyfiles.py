#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import autopath
import sys

from alex.corpustools.wavaskey import load_wavaskey, save_wavaskey


def main():

    files = []
    for i in range(1, len(sys.argv)):

        k = load_wavaskey(sys.argv[i], unicode)
        print sys.argv[i], len(k)
        files.append(k)

    keys = set()
    keys.update(set(files[0].keys()))
    ukeys = set()
    for f in files:
        keys = keys.intersection(set(f.keys()))
        ukeys = ukeys.union(set(f.keys()))

    print len(keys), len(ukeys), len(ukeys - keys)

    for f in files:
        rk = set(f.keys()) - keys
        for k in rk:
            if k in f:
                del f[k]

    for i in range(1, len(sys.argv)):
        save_wavaskey(sys.argv[i]+'.pruned',files[i-1])

if __name__ == '__main__':
    main()
