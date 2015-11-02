import glob
import os

import numpy as np
import pandas as pd

from lxml import etree


def parse_dates(filenames):
    for orig in filenames:
        *base, filename = orig.split("/")
        orig = filename[2:filename.index("_")].replace('--', '-')
        date = ""
        for c in orig:
            if c.isdigit() or c == "-":
                date += c
            else: break
        date = pd.datetime.strptime(date, "%Y-%m-%d" if date.count("-") == 2 else 
                                          "%Y-%m" if date.count("-") == 1 else 
                                           "%Y") 
        yield date        

def load_letters(folder):
    letter_filenames = glob.glob(os.path.join(folder, "*.txt"))
    letter_dates = np.array(list(parse_dates(letter_filenames)))
    return letter_filenames, letter_dates

def load_rrh(folder):
    rrh_filenames = glob.glob(folder + "/*/*-cleaned.txt")
    rrh_dates = []
    for idnumber in rrh_filenames:
        idnumber = idnumber.split('/')[-2]
        with open(folder + '/' + idnumber + '/META.xml') as infile:
            rrh_dates.append(int(etree.parse(infile).xpath("//year_estimate")[0].text))
    rrh_dates = pd.to_datetime(rrh_dates, format="%Y")
    return rrh_filenames, rrh_dates