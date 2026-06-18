import time
import logging
import warnings
import argparse
import itertools
import pandas as pd

from joblib import dump
from pathlib import Path
from rdkit.Chem import AllChem, CanonSmiles
from Admodule import Reader, Grouping, Utils
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

"""
Modified Preprocessing for 3-class TSV inputs.

Goal:
- Input: class1.tsv, class2.tsv, class3.tsv
- Build G1 from class1 only (size = --g1_size)
- Build G2 = (class1_remains + class2 + class3)
- Split G2 into train/valid/test with class-balanced 8:2:2 ratio
- Keep the rest (vectorize, scaling, saving) as original workflow
"""

logger = logging.getLogger(__name__)


def pick_molecules(df, cls_num, cores, rb=False):
    rb_idx = 0
    check = True
    rb_lst = [x for x in itertools.product([3, 2, 1], [2048, 1024, 512, 256])]
    clt = Grouping.Cluster()

    # handle edge cases
    if cls_num <= 0:
        return df.iloc[0:0].copy(), df.copy()
    if cls_num >= len(df):
        return df.copy(), df.iloc[0:0].copy()

    while check:
        if rb_idx == len(rb_lst) - 1:
            break
        logger.info(f'Use radius, nbits for Butina clustering : {rb_lst[rb_idx]}')
        cent, remains = clt.run(df, cls_num, rb_lst[rb_idx], cores)
        if len(cent) == cls_num:
            check = False
        rb_idx += 1

    if len(cent) > cls_num:
        cent.reset_index(drop=True, inplace=True)
        num_rows = len(cent) - cls_num
        random_rows = cent.sample(num_rows, random_state=42)

        cent.drop(random_rows.index, inplace=True)
        remains = pd.concat([remains, random_rows])
        remains.reset_index(drop=True, inplace=True)
    elif len(cent) < cls_num:
        cent.reset_index(drop=True, inplace=True)
        num_rows = cls_num - len(cent)
        random_rows = remains.sample(num_rows, random_state=42)

        remains.drop(random_rows.index, inplace=True)
        cent = pd.concat([cent, random_rows])
        cent.reset_index(drop=True, inplace=True)
    else:
        pass

    if rb:
        bit_generator = AllChem.GetMorganFingerprintAsBitVect
        cent['bits'] = cent['ROMol'].apply(
            lambda x: bit_generator(x, useChirality=True, radius=rb[0], nBits=rb[1])
        )
        remains['bits'] = remains['ROMol'].apply(
            lambda x: bit_generator(x, useChirality=True, radius=rb[0], nBits=rb[1])
        )
    return cent, remains


def duple_structures(df1, df2):
    df1['canonical_smi'] = df1['Smiles'].apply(CanonSmiles)
    df2['canonical_smi'] = df2['Smiles'].apply(CanonSmiles)
    remove_duple = df1[~df1['canonical_smi'].isin(df2['canonical_smi'])]
    return remove_duple


def _allocate_counts(n, train_u=8, valid_u=2, test_u=2):
    """Allocate counts for 8:2:2 while ensuring sum == n."""
    total_u = train_u + valid_u + test_u
    n_train = int(round(n * (train_u / total_u)))
    n_valid = int(round(n * (valid_u / total_u)))

    # ensure feasible bounds
    n_train = max(0, min(n_train, n))
    n_valid = max(0, min(n_valid, n - n_train))
    n_test = n - n_train - n_valid
    return n_train, n_valid, n_test


def split_by_class_balanced(df, cores, rb, train_u=8, valid_u=2, test_u=2):
    """
    Split df into train/valid/test with class-balanced allocation
    using pick_molecules per class.
    Requires df has 'Class' column.
    """
    train_parts, valid_parts, test_parts = [], [], []

    classes = sorted(df['Class'].unique())
    for cls in classes:
        sub = df[df['Class'] == cls].reset_index(drop=True)
        n = len(sub)
        n_train, n_valid, n_test = _allocate_counts(n, train_u, valid_u, test_u)

        if n == 0:
            continue

        train_c, remain = pick_molecules(sub, n_train, cores, rb)
        valid_c, remain2 = pick_molecules(remain, n_valid, cores, rb)
        # test = remaining (should match n_test)
        # to keep consistency with clustering-based picking, pick exactly n_test if possible
        if n_test > 0 and len(remain2) > 0:
            test_c, remain3 = pick_molecules(remain2, n_test, cores, rb)
        else:
            test_c = remain2.copy()

        train_parts.append(train_c)
        valid_parts.append(valid_c)
        test_parts.append(test_c)

        logger.info(
            f"[Class {cls}] total={n} -> train={len(train_c)}, valid={len(valid_c)}, test={len(test_c)}"
        )

    g2_train = pd.concat(train_parts).reset_index(drop=True) if train_parts else df.iloc[0:0].copy()
    g2_valid = pd.concat(valid_parts).reset_index(drop=True) if valid_parts else df.iloc[0:0].copy()
    g2_test = pd.concat(test_parts).reset_index(drop=True) if test_parts else df.iloc[0:0].copy()
    return g2_train, g2_valid, g2_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing data (3-class version)')
    parser.add_argument('--class1', required=True, help='Class1 data (tsv/sdf/etc supported by Custom_reader)')
    parser.add_argument('--class2', required=True, help='Class2 data')
    parser.add_argument('--class3', required=True, help='Class3 data')
    parser.add_argument('-o', '--output', type=str, required=True, help='Set your output path')

    # G1 size from class1
    parser.add_argument('--g1_size', type=int, default=30, help='How many molecules to pick from class1 as G1')

    # keep original options
    parser.add_argument('-r', '--radius', type=int, default=2, help='Set your radius')
    parser.add_argument('-b', '--bits', type=int, default=256, help='Set your nbits')
    parser.add_argument('-s', '--scaler', type=str, default='Standard', help='Set your scaler')
    parser.add_argument('-core', '--num_cores', type=int, default=2, help='Set the number of CPU cores to use')
    args = parser.parse_args()

    pd.set_option('mode.chained_assignment', None)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # path
    path_c1 = Path(args.class1)
    path_c2 = Path(args.class2)
    path_c3 = Path(args.class3)

    # file name base
    file_name = path_c1.stem.split('_')[0]
    path_output = Path(args.output) / f"{file_name}_preprocessing"
    path_output.mkdir(parents=True, exist_ok=True)

    # log
    Utils.set_log(path_output, 'preprocess.log')

    # Start
    start = time.time()

    # set cores
    n_cores = Utils.set_cores(args.num_cores)

    logger.info(f'Class1 data : {path_c1}')
    logger.info(f'Class2 data : {path_c2}')
    logger.info(f'Class3 data : {path_c3}')
    logger.info(f'Output path : {path_output}')
    logger.info(f'G1 size(from class1) : {args.g1_size}')
    logger.info(f'Fingerprint radius : {args.radius}')
    logger.info(f'Fingerprint nbits : {args.bits}')
    logger.info(f'Use cores : {n_cores}')
    logger.info("Split ratio (G2 only) : train:valid:test = 8:2:2 (class-balanced)")

    # load data
    c_reader = Reader.Custom_reader()
    cls1 = c_reader.run(path_c1.suffix, path_c1)
    cls2 = c_reader.run(path_c2.suffix, path_c2)
    cls3 = c_reader.run(path_c3.suffix, path_c3)

    # label classes
    cls1['Class'] = 1
    cls2['Class'] = 2
    cls3['Class'] = 3

    # pick G1 from class1 only
    rb = [args.radius, args.bits]
    g1, cls1_remain = pick_molecules(cls1, args.g1_size, n_cores, rb)

    g1['Group'] = 'G1'

    # G2 = everything else
    g2 = pd.concat([cls1_remain, cls2, cls3]).reset_index(drop=True)
    g2['Group'] = 'G2'

    logger.info(
        f"Counts -> class1={len(cls1)}, class2={len(cls2)}, class3={len(cls3)} | "
        f"G1={len(g1)}, G2={len(g2)}"
    )

    # split G2 into train/valid/test (class-balanced 8:2:2)
    g2_train, g2_valid, g2_test = split_by_class_balanced(g2, n_cores, rb, train_u=8, valid_u=2, test_u=2)

    logger.info(f"Final -> G1 : {len(g1)}, Train : {len(g2_train)}, Valid : {len(g2_valid)}, Test : {len(g2_test)}")

    # save dataset
    Utils.save(g1, path_output / file_name, custom="g1")
    Utils.save(g2_train, path_output / file_name, custom="train")
    Utils.save(g2_valid, path_output / file_name, custom="valid")
    Utils.save(g2_test, path_output / file_name, custom="test")

    # vectorize
    vectorize = Grouping.Vector()
    logger.info('Generate train vectors...')
    train_vector = vectorize.run(g1, g2_train, n_cores)
    logger.info('Generate valid vectors...')
    valid_vector = vectorize.run(g1, g2_valid, n_cores)
    logger.info('Generate test vectors...')
    test_vector = vectorize.run(g1, g2_test, n_cores)

    # save before scaling
    Utils.save(train_vector, path_output / file_name, custom="train_raw_vector")
    Utils.save(valid_vector, path_output / file_name, custom="valid_raw_vector")
    Utils.save(test_vector, path_output / file_name, custom="test_raw_vector")

    # scaling
    logger.info('Scaling vectors...')
    scalers = {'Robust': RobustScaler(), 'Standard': StandardScaler(), 'MinMax': MinMaxScaler()}
    scaler = scalers[args.scaler]

    cols = [col for col in train_vector.columns if col.startswith('f_')]
    train_vector[cols] = scaler.fit_transform(train_vector[cols])
    valid_vector[cols] = scaler.transform(valid_vector[cols])
    test_vector[cols] = scaler.transform(test_vector[cols])

    # save
    dump(scaler, path_output / f"{file_name}_{args.scaler}_scaler.pkl")
    Utils.save(train_vector, path_output / file_name, custom="train_vector")
    Utils.save(valid_vector, path_output / file_name, custom="valid_vector")
    Utils.save(test_vector, path_output / file_name, custom="test_vector")

    # finish
    runtime = time.time() - start
    logger.info(f"Time : {runtime}")
