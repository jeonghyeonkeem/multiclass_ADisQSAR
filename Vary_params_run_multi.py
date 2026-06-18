import sys
import time
import itertools
import subprocess
import multiprocessing
from joblib import dump
from pathlib import Path

import pandas as pd
from rdkit.Chem import PandasTools, AllChem
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from Admodule.Grouping import Cluster, Vector

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def to_float(x):
    if x is None:
        return float('nan')
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return float('nan')
    try:
        return float(s.split()[0])
    except Exception:
        return float('nan')



def fps(df, rd, bt, g1, n_cpu):
    PandasTools.AddMoleculeColumnToFrame(df)
    bit_generator = AllChem.GetMorganFingerprintAsBitVect

    df['bits'] = df['ROMol'].apply(
        lambda x: bit_generator(x, useChirality=True, radius=rd, nBits=bt)
    )
    g1['bits'] = g1['ROMol'].apply(
        lambda x: bit_generator(x, useChirality=True, radius=rd, nBits=bt)
    )

    vector = vectorize.run(g1, df, n_cpu)
    return vector


def pick_molecules(df, cls_num, cores):
    rb_idx = 0
    check = True
    rb_lst = [x for x in itertools.product([3, 2, 1], [2048, 1024, 512, 256])]
    clt = Cluster()

    while check:
        if rb_idx == len(rb_lst) - 1:
            break
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

    return cent, remains


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir

    base_out = script_dir / "Vary_params_results"
    base_out.mkdir(parents=True, exist_ok=True)

    scalers = {
        'Robust': RobustScaler(),
        'Standard': StandardScaler(),
        'MinMax': MinMaxScaler()
    }
    radius_type = {1: 'ECFP2', 2: 'ECFP4', 3: 'ECFP6'}

    clt = Cluster()
    vectorize = Vector()
    cores = max(1, multiprocessing.cpu_count() - 1)

    adis_qsar_py = script_dir / "ADis_QSAR_multi.py"
    if not adis_qsar_py.exists():
        raise FileNotFoundError(f"ADis_QSAR_multi.py not found: {adis_qsar_py}")

    prep_roots = sorted(data_path.glob("*_preprocessing"))
    if not prep_roots:
        raise FileNotFoundError(
            f"No '*_preprocessing' folders found under {data_path}"
        )

    for prep_root in prep_roots:
        g1_label = prep_root.stem.replace("_preprocessing", "")
        try:
            g1_cnt = int(g1_label)

        except ValueError:
            continue

        out_path = base_out / str(g1_cnt)
        out_path.mkdir(parents=True, exist_ok=True)

        summary_path = out_path / f"Summary_{g1_cnt}.tsv"

        done_keys = set()
        col_wr = True
        if summary_path.exists():
            df_done = pd.read_csv(summary_path, sep='\t')
            if {'Target', 'G1', 'Fingerprint_type', 'Scaler'}.issubset(df_done.columns):
                for _, row in df_done.iterrows():
                    key = (
                        str(row['Target']),
                        int(row['G1']),
                        str(row['Fingerprint_type']),
                        str(row['Scaler'])
                    )
                    done_keys.add(key)
                col_wr = False

        print(f"\nStart preprocessing set: G1 = {g1_cnt}")

        targets = [prep_root]
        for fd in targets:
            fn = fd.stem.replace("_preprocessing", "")
            print(f"\nStart target: {fn}")

            start = time.time()

            fdata_path = out_path
            fdata_path.mkdir(parents=True, exist_ok=True)

            g1 = pd.read_csv(fd / f"{fn}_g1.tsv", sep='\t')
            train = pd.read_csv(fd / f"{fn}_train.tsv", sep='\t')
            valid = pd.read_csv(fd / f"{fn}_valid.tsv", sep='\t')
            test = pd.read_csv(fd / f"{fn}_test.tsv", sep='\t')

            PandasTools.AddMoleculeColumnToFrame(g1)
            PandasTools.AddMoleculeColumnToFrame(train)
            PandasTools.AddMoleculeColumnToFrame(valid)
            PandasTools.AddMoleculeColumnToFrame(test)

            g1.to_csv(fdata_path / f"{fn}_g1.tsv", sep='\t', index=False)
            train.to_csv(fdata_path / f"{fn}_train.tsv", sep='\t', index=False)
            valid.to_csv(fdata_path / f"{fn}_valid.tsv", sep='\t', index=False)
            test.to_csv(fdata_path / f"{fn}_test.tsv", sep='\t', index=False)

            for radius in [1, 2, 3]:
                for nbits in [256, 512]:
                    train_vector = fps(train, radius, nbits, g1, cores)
                    valid_vector = fps(valid, radius, nbits, g1, cores)
                    test_vector = fps(test, radius, nbits, g1, cores)

                    fcols = [c for c in train_vector.columns if c.startswith("f_")]

                    for s_type, scaler in scalers.items():
                        key = (fn, g1_cnt, f"{radius_type[radius]}_{nbits}bits", s_type)
                        if key in done_keys:
                            continue

                        train_vector[fcols] = scaler.fit_transform(train_vector[fcols])
                        valid_vector[fcols] = scaler.transform(valid_vector[fcols])
                        test_vector[fcols] = scaler.transform(test_vector[fcols])

                        f_output = (
                            fdata_path / f"{radius_type[radius]}_{nbits}bits" / s_type
                        )
                        f_output.mkdir(parents=True, exist_ok=True)

                        dump(scaler, f_output / f"{fn}_{s_type}_scaler.pkl")
                        train_path = f_output / f"{fn}_train_vector.tsv"
                        valid_path = f_output / f"{fn}_valid_vector.tsv"
                        test_path = f_output / f"{fn}_test_vector.tsv"

                        train_vector.to_csv(train_path, sep='\t', index=False)
                        valid_vector.to_csv(valid_path, sep='\t', index=False)
                        test_vector.to_csv(test_path, sep='\t', index=False)

                        fwr = {
                            'Target': fn,
                            'G1': g1_cnt,
                            'Fingerprint_type': f"{radius_type[radius]}_{nbits}bits",
                            'Scaler': s_type
                        }

                        for md in ['RF', 'XGB', 'SVM', 'MLP']:
                            cmd = [
                                sys.executable,
                                str(adis_qsar_py),
                                '-train', str(train_path),
                                '-valid', str(valid_path),
                                '-test', str(test_path),
                                '-o', str(f_output),
                                '-m', md,
                                '-core', str(cores)
                            ]
                            stdout = subprocess.DEVNULL
                            subprocess.run(cmd, check=True)


                            model_path = f_output / f"{fn}_model" / md


                            mcs = pd.read_csv(
                                model_path / f"{fn}_{md}_model_score_log.tsv",
                                sep='\t'
                            )
                            ACC, AUC, PR, SP = [], [], [], []
                            for mcd in mcs.to_dict('records'):
                                name = mcd['Data'].capitalize()
                                ACC.append(f"{name} {to_float(mcd.get('ACC')):.2f}")
                                AUC.append(f"{name} {to_float(mcd.get('AUC_ovr_macro', mcd.get('AUC'))):.2f}")
                                PR.append(f"{name} {to_float(mcd.get('Precision_macro', mcd.get('Precision'))):.2f}")
                                RE.append(f"{name} {to_float(mcd.get('Recall_macro', mcd.get('Recall_macro'))):.2f}")

                            fwr[f"{md} ACC"] = " | ".join(ACC)
                            fwr[f"{md} AUC"] = " | ".join(AUC)
                            fwr[f"{md} PR"] = " | ".join(PR)
                            fwr[f"{md} RE"] = " | ".join(SP)

                        with open(summary_path, 'a', encoding='utf-8') as fw:
                            if col_wr:
                                fw.write('\t'.join(fwr.keys()) + '\n')
                                col_wr = False
                            fw.write('\t'.join(map(str, fwr.values())) + '\n')

            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
            print(f"Finished target: {fn} | time: {elapsed}")
