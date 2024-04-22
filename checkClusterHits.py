import pyLCIO # type: ignore
from pyLCIO import EVENT, UTIL

import itertools
import multiprocessing as mp
import os
import numpy as np
import pandas as pd

from typing import Any, List
from dataclasses import dataclass

encoding = 'system:0:5,side:5:-2,module:7:8,stave:15:4,layer:19:9,submodule:28:4,x:32:-16,y:48:-16'

def main():
    fnames = [
        '/data/fmeloni/DataMuC_MuColl10_v0A/v1/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10000.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/v1/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10100.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/v1/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10200.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/v1/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10300.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/v1/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10400.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/v1/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10500.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/v1/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10600.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/v1/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10700.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/v1/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10800.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/v1/reco/neutronGun_E_250_1000/neutronGun_E_250_1000_reco_10900.slcio',
    ]
    checker = CaloHitChecker(fnames)
    checker.readHits()


class names:
    clusters = 'PandoraClusters'
    ecal_barrel = 'EcalBarrelCollectionRec'
    ecal_endcap = 'EcalEndcapCollectionRec'
    hcal_barrel = 'HcalBarrelCollectionRec'
    hcal_endcap = 'HcalEndcapCollectionRec'
    yoke = 'MUON'

class systems:
    ecal_barrel = 20
    ecal_endcap = 29
    hcal_barrel = 10
    hcal_endcap = 11
    yoke_barrel = 13
    yoke_endcap = 14
    ecal = [ecal_barrel,
            ecal_endcap]
    hcal = [hcal_barrel,
            hcal_endcap]
    yoke = [yoke_barrel,
            yoke_endcap]


class CaloHitChecker:

    def __init__(self, fnames: List[str]) -> None:
        self.fnames = fnames
        self.collection_names = [getattr(names, attr) for attr in
                                 filter(lambda attr: not attr.startswith('_'), dir(names))]


    def readHits(self) -> None:
        n_workers = 1
        with mp.Pool(n_workers) as pool:
            list_of_dfs = pool.map(self.readHitsSerially, self.fnames)
            print('Merging ...')
            df = self.mergeDataFrames(list_of_dfs)
        print(f'Total:\n{df}')
        print(f'Total:\n{df.sum()}')


    def readHitsSerially(self, fname: str) -> List[pd.DataFrame]:
        print(f'Processing {fname} ...')
        reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
        reader.open(fname)
        reader.setReadCollectionNames(self.collection_names)
        return [self.processEventCellView(event) for event in reader]


    def processEventCellView(self, event: Any) -> pd.DataFrame:
        df = self.initDataFrame(1)
        decoder = UTIL.BitField64(encoding)
        if names.clusters not in event.getCollectionNames():
            return df
        col = event.getCollection(names.clusters)
        total = {'ecal hits-on-cluster': 0, 'hcal hits-on-cluster': 0, 'yoke hits-on-cluster': 0}
        for i_clus, clus in enumerate(col):
            for i_hit, hit in enumerate(clus.getCalorimeterHits()):
                cellid = int(hit.getCellID0() & 0xffffffff) | (int( hit.getCellID1() ) << 32)
                decoder.setValue(cellid)
                system = decoder['system'].value()
                ecal = system in systems.ecal
                hcal = system in systems.hcal
                yoke = system in systems.yoke
                if not (ecal ^ hcal) and not yoke:
                    raise Exception(f'Hit must be ecal ({ecal}) xor hcal ({hcal})')
                total['ecal hits-on-cluster'] += ecal
                total['hcal hits-on-cluster'] += hcal
                total['yoke hits-on-cluster'] += yoke
        df.loc[0] = total
        return df


    def printEncoding(self, event: Any) -> Any:
        for name in [
                names.ecal_barrel,
                names.ecal_endcap,
                names.hcal_barrel,
                names.hcal_endcap,
        ]:
            if name not in event.getCollectionNames():
                continue
            col = event.getCollection(name)
            encoding = col.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
            print(encoding)


    def mergeDataFrames(self, dfs: Any) -> pd.DataFrame:
        return pd.concat(itertools.chain.from_iterable(dfs), ignore_index=True)


    def initDataFrame(self, rows: int) -> pd.DataFrame:
        return pd.DataFrame({
            'ecal hits-on-cluster': np.zeros(rows, dtype=int),
            'hcal hits-on-cluster': np.zeros(rows, dtype=int),
            'yoke hits-on-cluster': np.zeros(rows, dtype=int),
        })

if __name__ == '__main__':
    main()

