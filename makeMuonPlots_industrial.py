import pyLCIO # type: ignore
import glob
import math
import os

import numpy as np
import pandas as pd # type: ignore
import matplotlib as mpl # type: ignore
mpl.use('Agg')
import matplotlib.pyplot as plt # type: ignore

from typing import List

def main() -> None:
    filepath = "/data/fmeloni/DataMuC_MuColl10_v0A/reco/muonGun_pT_0_50/*.slcio"
    fnames = glob.glob(filepath)
    df = getData(fnames)
    plot(df)


def plot(df: pd.DataFrame) -> None:
    print('Plotting ...')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), constrained_layout=True)
    bins = np.arange(0, 100, 1)
    ax[0].hist(df['pt'], bins=bins)
    ax[1].hist(df['pt'][df['ismuon']], bins=bins)
    ax[0].set_title('All PFO')
    ax[1].set_title('Muon PFO')
    for i_ax in range(len(ax)):
        ax[i_ax].set_xlabel('PFO $p_{T}$ [GeV]')
        ax[i_ax].set_ylabel('N(PFO)')
        ax[i_ax].tick_params(right=True, top=True)
    plt.savefig('industry.pdf')


def getData(fnames: List[str]) -> pd.DataFrame:
    # Read LCIO into pandas
    pqname = 'makeMuonPlots_industrial.parquet'
    pkname = 'makeMuonPlots_industrial.pickle'
    if os.path.exists(pqname):
        df = pd.read_parquet(pqname)
    else:
        df = processFiles(fnames)
        postProcess(df)
        print(df)
        print('Writing to file ...')
        # pip install pyarrow
        df.to_parquet(pqname)
        df.to_pickle(pkname)
    return df


def postProcess(df: pd.DataFrame) -> None:
    df['pt'] = np.linalg.norm(df[['px', 'py']], axis=1)
    df['ismuon'] = isMuon(df['type'])


def processFiles(fnames: List[str]) -> pd.DataFrame:
    n_rows = 0
    dfs = []
    for i_fname, f in enumerate(fnames):
        if i_fname % 100 == 0:
            print(f"Processing file {i_fname}/{len(fnames)}")
        dfs.append( processFile(f) )
        # if i_fname > 40:
        #     break
    return pd.concat(dfs, ignore_index=True)


def processFile(fname: str) -> pd.DataFrame:
    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(fname)
    return processEvents(reader)


def processEvents(reader) -> pd.DataFrame:
    n_rows = 0
    dfs = []
    for i_event, event in enumerate(reader):
        dfs.append( processEvent(event) )
    return pd.concat(dfs)


def processEvent(event) -> pd.DataFrame:
    pfos = event.getCollection("PandoraPFOs")
    n_rows = len(pfos)
    df = createDataframe(n_rows)
    for row, pfo in enumerate(pfos):
        pfo_p = pfo.getMomentum()
        df.at[row, 'px']   = pfo_p[0]
        df.at[row, 'py']   = pfo_p[1]
        df.at[row, 'pz']   = pfo_p[2]
        df.at[row, 'type'] = pfo.getType()
    return df


def createDataframe(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame({
        'px':     np.zeros(n_rows, dtype=float),
        'py':     np.zeros(n_rows, dtype=float),
        'pz':     np.zeros(n_rows, dtype=float),
        'type':   np.zeros(n_rows, dtype=int),
    })


def isMuon(val: int) -> int:
    return np.abs(val) == 13


if __name__ == '__main__':
    main()

