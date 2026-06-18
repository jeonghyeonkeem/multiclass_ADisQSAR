import logging
import numpy as np
import pandas as pd
import multiprocessing

from pathlib import Path
from rdkit.Chem import PandasTools


# module info option
logger = logging.getLogger(__name__)


def set_log(path_output, log_message):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(path_output / log_message),
            logging.StreamHandler()
        ]
    )


def set_cores(cores):
    max_cores = multiprocessing.cpu_count()
    if cores > max_cores:
        logger.info(f"max cpu cores is {max_cores} / Automatically set to max cores")
        n_cores = max_cores
    else:
        n_cores = cores
    return n_cores


def save(data, out, custom=None):
    import pandas as pd
    from pathlib import Path

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if custom is None:
        file_path = out.with_suffix(".tsv")
    else:
        file_path = Path(str(out) + f"_{custom}.tsv")

    # ---- 여기부터 추가/수정 ----
    # RDKit Mol 컬럼은 "Smiles" 컬럼이 있을 때만 만든다 (대소문자도 대응)
    if isinstance(data, pd.DataFrame):
        cols_lower = {c.lower(): c for c in data.columns}
        if "smiles" in cols_lower:
            try:
                from rdkit.Chem import PandasTools
                PandasTools.AddMoleculeColumnToFrame(data, smilesCol=cols_lower["smiles"])
            except Exception:
                pass
    # ---- 여기까지 ----

    data.to_csv(file_path, sep="\t", index=False)


