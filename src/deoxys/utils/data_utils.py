# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import pandas as pd


def read_csv(filepath, *args, **kwargs):
    # Load data to file
    df = pd.read_csv(filepath, *args, **kwargs)

    # Convert str-like array into float numbers
    object_keys = [key for key, dtype in df.dtypes.items()
                   if dtype == 'object']
    for key in object_keys:
        df[key] = df[key].map(_arraystr2float, na_action='ignore')

    return df


def _arraystr2float(val):
    if '"[' in val and ']"' in val:
        array = val[2: -2].split(',')
        if len(array) == 1:
            try:
                return float(array[0])
            except Exception:
                return array[0]
        else:
            return array
    else:
        return val
