#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import codecs
import os
import re
import sys

import autopath

from alex.utils.config import online_update, to_project_path

__all__ = ['database']


database = {
    "task": {
        "find_connection": ["najít spojení", "najít spoj", "zjistit spojení",
                            "zjistit spoj", "hledám spojení", 'spojení', 'spoj',
                           ],
        "find_platform": ["najít nástupiště", "zjistit nástupiště", ],
        'weather': ['počasí', ],
    },
    "time": {
        "now": ["nyní", "teď", "teďka", "hned", "nejbližší", "v tuto chvíli"],
        "0:01": ["minutu", ],
        "0:15": ["čtvrt hodiny", ],
        "0:30": ["půl hodiny", ],
        "0:45": ["tři čtvrtě hodiny", ],
        "1:00": ["hodinu", ],
    },
    "date_rel": {
        "today": ["dnes", "dneska",
                  "dnešek", "dneška", "dnešku", "dneškem",
                  "dnešní", "dnešnímu", "dnešního", "dnešním"],
        "tomorrow": ["zítra", "zejtra",
                     "zítřek", "zítřka", "zítřku", "zítřkem",
                     "zítřejší", "zítřejšímu", "zítřejším", "zítřejšího"],
        "day_after_tomorrow": ["pozítří", "pozejtří"],
    },
    "stop": {
    },
    "vehicle": {
        "bus": ["bus", "busem", "autobus", "autobusy", "autobusem", "autobusové"],
        "tram": ["tram", "tramvaj", "tramvajový", "tramvaje", "tramvají", "tramvajka", "tramvajkou", "šalina", "šalinou"],
        "metro": ["metro", "metrem", "metrema", "metru", "krtek", "krtkem", "podzemka", "podzemkou"],
        "train": ["vlak", "vlakem", "vlaky", "vlakovém", "rychlík", "rychlíky", "rychlíkem", "panťák", "panťákem"],
        "cable_car": ["lanovka", "lanovky", "lanovce", "lanovkou", "lanová dráha", "lanovou dráhou"],
        "ferry": ["přívoz", "přívozy", "přívozem", "přívozu", "loď", "lodí"],
    },
    "ampm": {
        "morning": ["ráno", "nadránem"],
        "am": ["dopo", "dopoledne", ],
        "pm": ["odpo", "odpoledne", ],
        "evening": ["večer", "podvečer", ],
        "night": ["noc", "noci"],
    },
    "city": {
    },
}

NUMBERS_1 = ["nula", "jedna", "dvě", "tři", "čtyři", "pět", "šest", "sedm",
             "osm", "devět", ]
NUMBERS_10 = ["", "deset", "dvacet", "třicet", "čtyřicet", "padesát",
              "šedesát", ]
NUMBERS_TEEN = ["deset", "jedenáct", "dvanáct", "třináct", "čtrnáct",
                "patnáct", "šestnáct", "sedmnáct", "osmnáct", "devatenáct"]
NUMBERS_ORD = ["nultý", "první", "druhý", "třetí", "čtvrtý", "pátý", "šestý",
               "sedmý", "osmý", "devátý", "desátý", "jedenáctý", "dvanáctý",
               "třináctý", "čtrnáctý", "patnáctý", "šestnáctý", "sedmnáctý",
               "osmnáctý", "devatenáctý", "dvacátý", "jednadvacátý",
               "dvaadvacátý", "třiadvacátý"]

# name of the file with one stop per line, assumed to reside in the same
# directory as this script
#
# The file is expected to have this format:
#   <value>; <phrase>; <phrase>; ...
# where <value> is the value for a slot and <phrase> is its possible surface
# form.
STOPS_FNAME = "stops.expanded.txt"
CITIES_FNAME = "cities.expanded.txt"

# load new stops & cities list from the server if needed
online_update(to_project_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), STOPS_FNAME)))
online_update(to_project_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), CITIES_FNAME)))


def db_add(category_label, value, form):
    """A wrapper for adding a specified triple to the database."""
#    category_label = category_label.strip()
#    value = value.strip()
#    form = form.strip()

    if len(value) == 0 or len(form) == 0:
        return

    if value in database[category_label] and isinstance(database[category_label][value], list):
        database[category_label][value] = set(database[category_label][value])

#    if category_label == 'stop':
#        if value in set(['Nová','Praga','Metra','Konečná','Nádraží',]):
#            return
            
#    for c in '{}+/&[],-':
#        form = form.replace(' %s ' % c, ' ')
#        form = form.replace(' %s' % c, ' ')
#        form = form.replace('%s ' % c, ' ')
#    form = form.strip()

    database[category_label].setdefault(value, set()).add(form)


# TODO allow "jednadvacet" "dvaadvacet" etc.
def spell_number(num):
    """Spells out the number given in the argument."""
    tens, units = num / 10, num % 10
    tens_str = NUMBERS_10[tens]
    units_str = NUMBERS_1[units]
    if tens == 1:
        return NUMBERS_TEEN[units]
    elif tens:
        if units:
            return "{t} {u}".format(t=tens_str, u=units_str)
        return "{t}".format(t=tens_str)
    else:
        return units_str


def add_time():
    """
    Basic approximation of all known explicit time expressions.

    Handles:
        <hour>
        <hour> hodin(a/y)
        <hour> hodin(a/y) <minute>
        <hour> <minute>
        půl/čtvrt/tři čtvrtě <hour>
        <minute> minut(u/y)
    where <hour> and <minute> are spelled /given as numbers.

    Cannot yet handle:
        za pět osm
        dvacet dvě hodiny
    """
    # ["nula", "jedna", ..., "padesát devět"]
    numbers_str = [spell_number(num) for num in xrange(60)]
    hr_id_stem = 'hodin'
    hr_endings = {1: [('u', 'u'), ('a', 'a')],
                  2: [('y', '')], 3: [('y', '')], 4: [('y', '')]}

    min_id_stem = 'minut'
    min_endings = {1: [('u', 'u'), ('a', 'a')],
                   2: [('y', '')], 3: [('y', '')], 4: [('y', '')]}

    for hour in xrange(24):
        # set stems for hours (cardinal), hours (ordinal)
        hr_str_stem = numbers_str[hour]
        if hour == 22:
            hr_str_stem = 'dvacet dva'
        hr_ord = NUMBERS_ORD[hour]
        if hr_ord.endswith('ý'):
            hr_ord = hr_ord[:-1] + 'é'
        if hour == 1:
            hr_ord = 'jedné'
            hr_str_stem = 'jedn'

        # some time expressions are not declined -- use just 1st ending
        _, hr_str_end = hr_endings.get(hour, [('', '')])[0]
        # X:00
        add_db_time(hour, 0, "{ho} hodině", {'ho': hr_ord})

        if hour >= 1 and hour <= 12:
            # (X-1):15 quarter past (X-1)
            add_db_time(hour - 1, 15, "čtvrt na {h}",
                        {'h': hr_str_stem + hr_str_end})
            # (X-1):30 half past (X-1)
            add_db_time(hour - 1, 30, "půl {ho}", {'ho': hr_ord})
            # (X-1):45 quarter to X
            add_db_time(hour - 1, 45, "tři čtvrtě na {h}",
                        {'h': hr_str_stem + hr_str_end})

        # some must be declined (but variants differ only for hour=1)
        for hr_id_end, hr_str_end in hr_endings.get(hour, [('', '')]):
            # X:00
            add_db_time(hour, 0, "{h}", {'h': hr_str_stem + hr_str_end})
            add_db_time(hour, 0, "{h} {hi}", {'h': hr_str_stem + hr_str_end,
                                              'hi': hr_id_stem + hr_id_end})
            # X:YY
            for minute in xrange(60):
                min_str = numbers_str[minute]
                add_db_time(hour, minute, "{h} {hi} {m}",
                            {'h': hr_str_stem + hr_str_end,
                             'hi': hr_id_stem + hr_id_end, 'm': min_str})
                add_db_time(hour, minute, "{h} {hi} a {m}",
                            {'h': hr_str_stem + hr_str_end,
                             'hi': hr_id_stem + hr_id_end, 'm': min_str})
                if minute < 10:
                    min_str = 'nula ' + min_str
                add_db_time(hour, minute, "{h} {m}",
                            {'h': hr_str_stem + hr_str_end, 'm': min_str})

    # YY minut(u/y)
    for minute in xrange(60):
        min_str_stem = numbers_str[minute]
        if minute == 22:
            min_str_stem = 'dvacet dva'
        if minute == 1:
            min_str_stem = 'jedn'

        for min_id_end, min_str_end in min_endings.get(minute, [('', '')]):
            add_db_time(0, minute, "{m} {mi}", {'m': min_str_stem + min_str_end,
                                                'mi': min_id_stem + min_id_end})


def add_db_time(hour, minute, format_str, replacements):
    """Add a time expression to the database
    (given time, format string and all replacements as a dict)."""
    time_val = "%d:%02d" % (hour, minute)
    db_add("time", time_val, format_str.format(**replacements))


def preprocess_cl_line(line):
    """Process one line in the category label database file."""
    name, forms = line.strip().split("\t")
    forms = [form.strip() for form in forms.split(';')]
    return name, forms


def add_from_file(category_label, fname):
    """Adds to the database names + surface forms of all category labels listed in the given file.
    The file must contain the category lablel name + tab + semicolon-separated surface forms on each
    line.
    """
    dirname = os.path.dirname(os.path.abspath(__file__))
    with codecs.open(os.path.join(dirname, fname), encoding='utf-8') as stops_file:
        for line in stops_file:
            if line.startswith('#'):
                continue
            val_name, val_surface_forms = preprocess_cl_line(line)
            for form in val_surface_forms:
                db_add(category_label, val_name, form)


def add_stops():
    """Add stop names from the stops file."""
    add_from_file('stop', STOPS_FNAME)


def add_cities():
    """Add city names from the cities file."""
    add_from_file('city', CITIES_FNAME)


def save_c2v2f(file_name):
    c2v2f = []
    for k in database:
        for v in database[k]:
            for f in database[k][v]:
                if re.search('\d', f):
                    continue
                c2v2f.append((k, v, f))

    c2v2f.sort()

    # save the database vocabulary - all the surface forms
    with codecs.open(file_name, 'w', 'UTF-8') as f:
        for x in c2v2f:
            f.write(' => '.join(x))
            f.write('\n')

def save_surface_forms(file_name):
    surface_forms = []
    for k in database:
        for v in database[k]:
            for f in database[k][v]:
                if re.search('\d', f):
                    continue
                surface_forms.append(f)
    surface_forms.sort()

    # save the database vocabulary - all the surface forms
    with codecs.open(file_name, 'w', 'UTF-8') as f:
        for sf in surface_forms:
            f.write(sf)
            f.write('\n')


def save_SRILM_classes(file_name):
    surface_forms = []
    for k in database:
        for v in database[k]:
            for f in database[k][v]:
                if re.search('\d', f):
                    continue
                surface_forms.append("CL_" + k.upper() + " " + f.upper())
    surface_forms.sort()

    # save the database vocabulary - all the surface forms
    with codecs.open(file_name, 'w', 'UTF-8') as f:
        for sf in surface_forms:
            f.write(sf)
            f.write('\n')

########################################################################
#                  Automatically expand the database                   #
########################################################################
add_time()
add_stops()
add_cities()

if "dump" in sys.argv or "--dump" in sys.argv:
    save_c2v2f('database_c2v2f.txt')
    save_surface_forms('database_surface_forms.txt')
    save_SRILM_classes('database_SRILM_classes.txt')
