import pyLCIO # type: ignore
from pyLCIO import EVENT, UTIL

import argparse
import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd
import time

from typing import Any, List, Tuple


def options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(usage=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", help="Comma-separated input filenames")
    return parser.parse_args()


def main() -> None:
    ops = options()
    if not ops.i:
        raise Exception('Need input file with -i')
    fnames = ops.i.split(',')
    # fnames = [
    #     '/data/fmeloni/DataMuC_MuColl10_v0A/reco/pionGun_pT_250_1000/pionGun_pT_250_1000_reco_4280.slcio',
    #     '/data/fmeloni/DataMuC_MuColl10_v0A/reco/pionGun_pT_250_1000/pionGun_pT_250_1000_reco_4290.slcio',
    # ]
    pqname = 'writeCaloHits.parquet'
    writer = CaloHitWriter(fnames, pqname)
    writer.readHits()
    # writer.writeHits()


class CaloHitWriter:

    def __init__(self, fnames: List[str], pqname: str) -> None:
        self.fnames = fnames
        self.pqname = pqname


    def readHits(self) -> None:
        n_workers = 10
        list_of_dfs = []
        with mp.Pool(n_workers) as pool:
            list_of_dfs = pool.map(self.readHitsSerially, self.fnames)
        self.df = self.mergeDataFrames(list_of_dfs)


    def readHitsSerially(self, fname: str) -> List[pd.DataFrame]:
        reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
        reader.open(fname)
        start = time.perf_counter()
        dfs = [self.processEventCellView(event) for event in reader]
        end = time.perf_counter()
        self.announceTime(end-start, len(dfs))
        return dfs


    def announceTime(self, duration: float, n: int) -> None:
        rate = n/duration
        print(f'Event rate: {rate:.3f} Hz ({n} events)')


    def processEventCellView(self, event: Any) -> pd.DataFrame:
        # print(f'On event {event}')
        colnames = [
            'HCalBarrelCollection',
            'HCalEndcapCollection',
        ]
        truth_px, truth_py, truth_pz, truth_e = self.processEventTruth(event)
        cols = [event.getCollection(name) for name in colnames]
        d = {
            'event': [],
            'hit_system': [],
            'hit_side': [],
            'hit_layer': [],
            'hit_x': [],
            'hit_y': [],
            'hit_z': [],
            'hit_e': [],
            'truth_px': [],
            'truth_py': [],
            'truth_pz': [],
            'truth_e': [],
        }
        for colname, col in zip(colnames, cols):
            cellIdEncoding = col.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
            cellIdDecoder = UTIL.BitField64(cellIdEncoding)
            for i_hit, hit in enumerate(col):
                cellIdDecoder.setValue((hit.getCellID0() & 0xffffffff) |
                                       (hit.getCellID1() << 32))
                x, y, z = self.getPosition(hit)
                d['event'].append(event.getEventNumber())
                d['hit_system'].append(cellIdDecoder['system'].value())
                d['hit_side'].append(cellIdDecoder['side'].value())
                d['hit_layer'].append(cellIdDecoder['layer'].value())
                d['hit_x'].append(x)
                d['hit_y'].append(y)
                d['hit_z'].append(z)
                d['hit_e'].append(hit.getEnergy())
                d['truth_px'].append(truth_px)
                d['truth_py'].append(truth_py)
                d['truth_pz'].append(truth_pz)
                d['truth_e'].append(truth_e)
                # if i_hit < 10:
                #     print(i_hit, cellIdDecoder.valueString(), f'x={x:.1f} y={y:.1f} z={z:.1f}')
        return pd.DataFrame(d)


    def processEventTruth(self, event: Any) -> Tuple[float, float, float, float]:
        colname = 'MCParticle'
        event_number = event.getEventNumber()
        n_stable = 0
        for obj in event.getCollection(colname):
            if obj.getGeneratorStatus() == 1:
                obj_p = obj.getMomentum()
                obj_e = obj.getEnergy()
                pxypyze = obj_p[0], obj_p[1], obj_p[2], obj_e
                n_stable += 1
                # break
        if n_stable != 1:
            raise Exception('Unexpected truth particles')
        return pxypyze


    def getPosition(self, hit: Any) -> Tuple[float, float, float]:
        NDATA = 3
        return tuple(np.frombuffer(hit.getPosition(), dtype=np.float32, count=NDATA))


    def mergeDataFrames(self, dfs: List[List[pd.DataFrame]]) -> pd.DataFrame:
        return pd.concat(itertools.chain.from_iterable(dfs), ignore_index=True)


    def writeHits(self):
        if self.df is None:
            raise Exception('Empty DataFrame')
        print(f'Writing hits to file: {len(self.df)}')
        self.df.to_parquet(self.pqname)
        

if __name__ == '__main__':
    main()

