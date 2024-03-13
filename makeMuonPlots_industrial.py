import pyLCIO
import glob
import math
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def main():

    # Gather input files
    filepath = "/data/fmeloni/DataMuC_MuColl10_v0A/reco/muonGun_pT_0_50/*.slcio"
    fnames = glob.glob(filepath)

    # Read LCIO into pandas
    pqname = 'makeMuonPlots_industrial.parquet'
    pkname = 'makeMuonPlots_industrial.pickle'
    if False and os.path.exists(pqname):
        df = pd.read_parquet(pqname)
    else:
        df = processFiles(fnames)
        postProcess(df)
        print(df)
        print('Writing to file ...')
        df.to_parquet(pqname)
        df.to_pickle(pkname)

    # Plot
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

def postProcess(df):
    df['pt'] = np.linalg.norm(df[['px', 'py']], axis=1)

def processFiles(fnames):
    n_rows = 0
    dfs = []
    for i_fname, f in enumerate(fnames):
        if i_fname % 100 == 0:
            print(f"Processing file {i_fname}/{len(fnames)}")
        dfs.append( processFile(f) )
        # if i_fname > 40:
        #     break
    return pd.concat(dfs, ignore_index=True)

def processFile(fname):
    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(fname)
    return processEvents(reader)

def processEvents(reader):
    n_rows = 0
    dfs = []
    for i_event, event in enumerate(reader):
        dfs.append( processEvent(event) )
    return pd.concat(dfs)

def processEvent(event):
    pfos = event.getCollection("PandoraPFOs")
    n_rows = len(pfos)
    df = createDataframe(n_rows)
    for row, pfo in enumerate(pfos):
        pfo_p = pfo.getMomentum()
        df.at[row, 'px'] = pfo_p[0]
        df.at[row, 'py'] = pfo_p[1]
        df.at[row, 'type'] = pfo.getType()
        df.at[row, 'ismuon'] = isMuon(pfo.getType())
    return df

def createDataframe(n_rows):
    return pd.DataFrame({
        # 'pt':     np.zeros(n_rows, dtype=float),
        'px':     np.zeros(n_rows, dtype=float),
        'py':     np.zeros(n_rows, dtype=float),
        'type':   np.zeros(n_rows, dtype=int),
        'ismuon': np.zeros(n_rows, dtype=bool),
    })

def isMuon(val):
    return np.abs(val) == 13

if __name__ == '__main__':
    main()

