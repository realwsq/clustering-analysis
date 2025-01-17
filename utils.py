import os, pickle
import pandas as pd

def remove_space(s):
    s = s.replace(" ","")
    s = s.replace("'", "")
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace("{", "")
    s = s.replace("}", "")
    s = s.replace(":", "")
    s = s.replace(",", "_")
    return s

def log_kv(**kwargs):
    print(f"{kwargs}")

def make_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder


def load_or_save_dict(_fname, _main, **params):
    if os.path.isfile(_fname):
        with open(_fname, 'rb') as f:
            res = pickle.load(f)
    else:
        res = _main(**params)
        with open(_fname, 'wb') as f:
            pickle.dump(res, f)
    return res

# * cells should not contain nparray
def load_or_save_df(_fname, _main, **params):
    if os.path.isfile(_fname):
        df = pd.read_csv(_fname)
    else:
        df = _main(**params)
        df.to_csv(_fname, index=False,)
    return df
