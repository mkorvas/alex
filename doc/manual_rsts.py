#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This code is PEP8-compliant. See http://www.python.org/dev/peps/pep-0008.

import os.path
import shutil
import fnmatch

root = '../alex'


def get_all_rsts(root='.'):
    """ Searches for all rst files under the root directory and the alex subdirectory.

    At this moment, there files are not sorted correctly as files from deeper directories can appear
    before the files fro upper directories.
    The solution would be to list first the files in a directory and after that files in the subdirectories and so on
    I will have to find a better way of listing files in subdirectories, may be ``walk``?

    :param root: the root directory form where the search begin
    :return: the sorted list of names of rst files
    """
    f, glob = set([]), '*.rst'
    for root, dirnames, filenames in os.walk(root, followlinks=False):
        for filename in fnmatch.filter(filenames, glob):
            f.add(os.path.join(root, filename))
    f = list(f)
    # f.sort()

    for i in f:
        shutil.copy2(i, os.path.join('_man_rst', i.replace('/', '.').replace('...', '')))

    return f

rsts = [os.path.join('..', f) for f in ['README.rst', 'INSTALL.rst']]
rsts.extend(get_all_rsts(root))

with open(os.path.join('_man_rst', 'index.rst'), 'w') as f:
    f.write('Index of manually written in source code tree documentation\n')
    f.write('===========================================================\n')
    f.write('.. toctree::\n')
    f.write('   :maxdepth: 2\n')
    f.write('   \n')
    for i in rsts:
        if 'alex' in i:
            f.write('   %s\n' % i.replace('/', '.').replace('...', '').replace('.rst', ''))
