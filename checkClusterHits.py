import pyLCIO # type: ignore
from pyLCIO import EVENT, UTIL

import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd # type: ignore

from typing import Any, List
from dataclasses import dataclass

encoding = 'system:0:5,side:5:-2,module:7:8,stave:15:4,layer:19:9,submodule:28:4,x:32:-16,y:48:-16'

def main():
    fnames = [
        '/data/fmeloni/DataMuC_MuColl10_v0A/reco/pionGun_pT_250_1000/pionGun_pT_250_1000_reco_4200.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/reco/pionGun_pT_250_1000/pionGun_pT_250_1000_reco_4210.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/reco/pionGun_pT_250_1000/pionGun_pT_250_1000_reco_4220.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/reco/pionGun_pT_250_1000/pionGun_pT_250_1000_reco_4230.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/reco/pionGun_pT_250_1000/pionGun_pT_250_1000_reco_4240.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/reco/pionGun_pT_250_1000/pionGun_pT_250_1000_reco_4250.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/reco/pionGun_pT_250_1000/pionGun_pT_250_1000_reco_4260.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/reco/pionGun_pT_250_1000/pionGun_pT_250_1000_reco_4270.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/reco/pionGun_pT_250_1000/pionGun_pT_250_1000_reco_4280.slcio',
        '/data/fmeloni/DataMuC_MuColl10_v0A/reco/pionGun_pT_250_1000/pionGun_pT_250_1000_reco_4290.slcio',
    ]
    checker = CaloHitChecker(fnames)
    checker.readHits()


class names:
    clusters = 'PandoraClusters'
    ecal_barrel = 'EcalBarrelCollectionRec'
    ecal_endcap = 'EcalEndcapCollectionRec'
    hcal_barrel = 'HcalBarrelsCollectionRec'
    hcal_endcap = 'HcalEndcapsCollectionRec'
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
        print('Total:\n', df)
        print('Total:\n', df.sum())


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
        total = {'ecal': 0, 'hcal': 0, 'yoke': 0}
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
                total['ecal'] += ecal
                total['hcal'] += hcal
                total['yoke'] += yoke
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
            'ecal': np.zeros(rows, dtype=int),
            'hcal': np.zeros(rows, dtype=int),
            'yoke': np.zeros(rows, dtype=int),
        })

if __name__ == '__main__':
    main()

