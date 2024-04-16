import pyLCIO
from pyLCIO import EVENT, UTIL

import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd

from typing import Any, List, Tuple

def main():
    fnames = [
        '/data/fmeloni/DataMuC_MuColl10_v0A/reco/pionGun_pT_250_1000/pionGun_pT_250_1000_reco_4280.slcio',
    ]
    pqname = 'writeCaloHits.parquet'
    writer = CaloHitWriter(fnames, pqname)
    writer.readHits()
    writer.writeHits()


class CaloHitWriter:

    def __init__(self, fnames: List[str], pqname: str) -> None:
        self.fnames = fnames
        self.pqname = pqname
        self.df = None


    def readHits(self) -> None:
        n_workers = 10
        list_of_dfs = []
        with mp.Pool(n_workers) as pool:
            list_of_dfs = pool.map(self.readHitsSerially, self.fnames)
        self.df = self.mergeDataFrames(list_of_dfs)


    def readHitsSerially(self, fname: str) -> List[pd.DataFrame]:
        reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
        reader.open(fname)
        return [self.processEventCellView(event) for event in reader]


    def processEventCellView(self, event: Any) -> pd.DataFrame:
        print(f'On event {event}')
        colnames = [
            'HCalBarrelCollection',
            'HCalEndcapCollection',
        ]
        cols = [event.getCollection(name) for name in colnames]
        rows = sum(len(col) for col in cols)
        df = self.initDataFrame(rows)
        row = 0
        for colname, col in zip(colnames, cols):
            # this might be slow as hell
            cellIdEncoding = col.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
            cellIdDecoder = UTIL.BitField64(cellIdEncoding)
            for i_hit, hit in enumerate(col):
                cellIdDecoder.setValue((hit.getCellID0() & 0xffffffff) |
                                       (hit.getCellID1() << 32))
                x, y, z = self.getPosition(hit)
                df.at[row, 'event']   = event.getEventNumber()
                df.at[row, 'hit_system'] = cellIdDecoder['system'].value()
                df.at[row, 'hit_side']   = cellIdDecoder['side'].value()
                df.at[row, 'hit_layer']  = cellIdDecoder['layer'].value()
                df.at[row, 'hit_x']   = x
                df.at[row, 'hit_y']   = y
                df.at[row, 'hit_z']   = z
                df.at[row, 'hit_e']   = hit.getEnergy()
                row += 1
                # if i_hit < 10:
                #     print(i_hit, cellIdDecoder.valueString(), f'x={x:.1f} y={y:.1f} z={z:.1f}')
        return df

    def processEvent(self, event: Any) -> pd.DataFrame:
        print(f'On event {event}')
        colname = 'PandoraClusters'
        col = event.getCollection(colname)
        rows = sum([len(ele.getCalorimeterHits()) for ele in col])
        print([len(ele.getCalorimeterHits()) for ele in col])       
        df = self.initDataFrame(rows)

        for name in ['ECalBarrelCollection',
                     'ECalEndcapCollection',
                     'HCalBarrelCollection',
                     'HCalEndcapCollection',
                     # 'EcalBarrelCollectionDigi',
                     'EcalEndcapCollectionDigi',
                     'EcalEndcapCollectionRec',
                     # 'HcalBarrelsCollectionDigi',
                     'HcalEndcapsCollectionDigi',
                     'HcalEndcapsCollectionRec',
                     ]:
            col_ = event.getCollection(name)
            # print(f'{name} -> {len(col_)}')

        # cellIdEncoding = 'M:3,S-1:3,I:9,J:9,K-1:6' # col.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
        # cellIdEncoding = "system:5,side:-2,module:8,stave:4,layer:9,submodule:4,x:32:-16,y:-16"
        cellIdEncoding = 'system:0:5,side:5:-2,module:7:8,stave:15:4,layer:19:9,submodule:28:4,x:32:-16,y:48:-16'
        cellIdDecoder = UTIL.BitField64(cellIdEncoding)


        for name in ['EcalEndcapCollectionRec',
                     'HcalEndcapsCollectionRec',
                     ]:
            col_ = event.getCollection(name)
            for i_hit, hit in enumerate(col_):
                if i_hit > 20:
                    continue
                if i_hit == 0:
                    print(col_.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding))
                cellId = int(hit.getCellID0() & 0xffffffff) | (int( hit.getCellID1() ) << 32)
                cellIdDecoder.setValue(cellId)
                x, y, z = self.getPosition(hit)
                print(i_hit, f'0x{cellId:08x}', cellIdDecoder.valueString(), f'x={x:.1f} y={y:.1f} z={z:.1f}')


        row = 0
        for i_ele, ele in enumerate(col):
            for i_hit, hit in enumerate(ele.getCalorimeterHits()):

                # print(i_ele, i_hit)

                if i_ele == 0 and False: # and i_hit < 2000:
                    cellId = int(hit.getCellID0() & 0xffffffff) | (int( hit.getCellID1() ) << 32)
                    cellIdDecoder.setValue(cellId)
                    # print(f'0x{cellIdDecoder.getValue():08x}')
                    print(rows, i_hit, f'0x{cellId:016x}', cellIdDecoder.valueString())
                    # print(cellIdDecoder.fieldDescription())
                    # side = int(cellIdDecoder['side'].value())
                    # layer = int(cellIdDecoder['layer'].value())
                    # print(len(cellIdEncoding), EVENT.LCIO.CellIDEncoding)

                x, y, z = self.getPosition(hit)
                df.at[row, 'event']   = event.getEventNumber()
                df.at[row, 'cluster'] = i_ele
                df.at[row, 'is_hcal'] = hit.getType()
                df.at[row, 'hit_x']   = x
                df.at[row, 'hit_y']   = y
                df.at[row, 'hit_z']   = z
                df.at[row, 'hit_e']   = hit.getEnergy()
                row += 1
        return df


    def getPosition(self, hit: Any) -> Tuple[float, float, float]:
        NDATA = 3
        return np.frombuffer(hit.getPosition(), dtype=np.float32, count=NDATA)


    def mergeDataFrames(self, dfs: List[List[pd.DataFrame]]) -> pd.DataFrame:
        return pd.concat(itertools.chain.from_iterable(dfs), ignore_index=True)


    def initDataFrame(self, rows: int):
        return pd.DataFrame({
            'event':      np.zeros(rows, dtype=int),
            'cluster':    np.zeros(rows, dtype=int),
            'is_hcal':    np.zeros(rows, dtype=bool),
            'hit_system': np.zeros(rows, dtype=int),
            'hit_side':   np.zeros(rows, dtype=int),
            'hit_layer':  np.zeros(rows, dtype=int),
            'hit_x':      np.zeros(rows, dtype=float),
            'hit_y':      np.zeros(rows, dtype=float),
            'hit_z':      np.zeros(rows, dtype=float),
            'hit_e':      np.zeros(rows, dtype=float),
        })


    def writeHits(self):
        if self.df is None:
            raise Exception('Empty DataFrame')
        self.df.to_parquet(self.pqname)
        

if __name__ == '__main__':
    main()

# reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
# reader.open(filepath)
# event = reader.readNextEvent()
# event = reader.readNextEvent()
# event = reader.readNextEvent()
# # event = reader.readNextEvent()


# def getMomentum(particle):
#     NDATA = 3
#     return np.frombuffer(particle.getMomentum(), dtype=np.float64, count=NDATA)

# def getPosition(hit):
#     NDATA = 3
#     return np.frombuffer(hit.getPosition(), dtype=np.float32, count=NDATA)

# def blurb(name):
#     if name not in event.getCollectionNames():
#         print(f'Skipping {name}')
#         return
#     col = event.getCollection(name)
#     n_ele = col.getNumberOfElements()
#     for i_ele, ele in enumerate(col):
#         def cell(hit):
#             try:
#                 return f'{hit.getCellID1():08x}.{hit.getCellID0():08x}'
#             except:
#                 return ''
#         def position(hit):
#             pos = (f'{hit.getPosition()[i]:.2f}' for i in range(3))
#             return ','.join(pos)
#         extra = ''
#         if 'Relations' in name:
#             extra = f'\n from={ele.getFrom()} \n to={ele.getTo()} \n weight={ele.getWeight()}'
#         print(name, f'{i_ele}/{n_ele}', ele, extra)
#     print()
#     # ele = None if col.getNumberOfElements() == 0 else col.getElementAt(0)
#     # print(name, col.getNumberOfElements(), ele)

# # blurb('HCalBarrelCollection')
# # blurb('HcalBarrelsCollectionDigi')
# # blurb('HcalBarrelsCollectionRec')
# # blurb('HcalBarrelsRelationsSimDigi')
# # blurb('HcalBarrelsRelationsSimRec')

# def compareEnergy(name):
#     col = event.getCollection(name)
#     n_ele = col.getNumberOfElements()
#     for i_ele, ele in enumerate(col):
#         if i_ele >= 10:
#             break
#         energy_from = ele.getFrom().getEnergy()
#         energy_to   = ele.getTo().getEnergy()
#         print(f'{name} {i_ele}/{n_ele}:  {energy_from:.6f} <-> {energy_to:.6f}')
#     print()

# # compareEnergy('HcalBarrelsRelationsSimDigi')
# # compareEnergy('HcalBarrelsRelationsSimRec')

# # for name in ['MCParticle',
# #              'MCPhysicsParticles',
# #              'HCalBarrelCollection',
# #              'HcalBarrelsCollectionDigi',
# #              'HcalBarrelsCollectionRec',
# #              'HCalEndcapCollection',
# #              'HcalEndcapsCollectionDigi',
# #              'HcalEndcapsCollectionRec',
# #              'PandoraClusters',
# #              'PandoraPFOs',
# #              'JetOut',
# #          ]:
# #     col = event.getCollection(name)
# #     n_ele = col.getNumberOfElements()
# #     if n_ele > 0:
# #         pass # print(n_ele, col[0], name)

# def theta(x, y, z):
#     return np.arccos(z / np.linalg.norm([x, y, z]))

# def phi(x, y):
#     return np.arctan2(y, x)

# name = 'PandoraClusters'
# print(name)
# col = event.getCollection(name)
# for it, ele in enumerate(col):
#     #print(ele)
#     n_hits = len(ele.getCalorimeterHits())
#     df = pd.DataFrame({
#         'hit_x': np.zeros(n_hits),
#         'hit_y': np.zeros(n_hits),
#         'hit_z': np.zeros(n_hits),
#         'hit_e': np.zeros(n_hits),
#     })
#     print(n_hits)
#     print(f'{it} E,theta,phi = {ele.getEnergy():.1f},{ele.getITheta():.3f},{ele.getIPhi():.3f}')
#     with open('tmp.txt', 'w') as fi:
#         for ih, hit in enumerate(ele.getCalorimeterHits()):
#             x, y, z = getPosition(hit)
#             df.at[ih, 'hit_x'] = x
#             df.at[ih, 'hit_y'] = y
#             df.at[ih, 'hit_z'] = z
#             df.at[ih, 'hit_e'] = hit.getEnergy()
#             fi.write(f' Hit {ih} x={x:5.1f} y={y:5.1f} z={z:5.1f} E={hit.getEnergy():.6f}\n')
#             if ih % 1000 == 0:
#                 print(f'On {ih} / {n_hits}')
#             if False and ih > 1000:
#                 print(' Stopping')
#                 break
#     print('Stopping after first cluster')
#     # df.to_parquet('play.parquet')
#     break

# print('*'*10)

# name = 'PandoraPFOs'
# print(name)
# col = event.getCollection(name)
# for it, ele in enumerate(col):
#     # print(ele)
#     px, py, pz = getMomentum(ele)
#     pn = np.linalg.norm([px, py, pz])
#     theta_, phi_ = theta(px, py, pz), phi(px, py)
#     #print(len(ele.getClusters()), len(ele.getParticles()))
#     print(f'{it} px={px:.1f} py={py:.1f} pz={pz:.1f} pn={pn:.1f} theta={theta_:.3f} phi={phi_:.3f}')
#     for track in ele.getTracks():
#         print(f' Track phi = {track.getPhi():.3f}')
#     #for clus in ele.getClusters():
#     #    print(f' Cluster {clus}')
# print('*'*10)

# name = 'JetOut'
# print(name)
# col = event.getCollection(name)
# for it, ele in enumerate(col):
#     #print(ele)
#     px, py, pz = getMomentum(ele)
#     pn = np.linalg.norm([px, py, pz])
#     theta_, phi_ = theta(px, py, pz), phi(px, py)
#     #print(len(ele.getClusters()), len(ele.getParticles()))
#     print(f'{it} px={px:.1f} py={py:.1f} pz={pz:.1f} pn={pn:.1f} theta={theta_:.3f} phi={phi_:.3f}')
#     #for part in ele.getParticles():
#     #    print(f' Particle {part}')
# print('*'*10)

# name = 'MCParticle'
# print(name)
# col = event.getCollection(name)
# for it, ele in enumerate(col):
#     if ele.getGeneratorStatus() != 1:
#         continue
#     px, py, pz = getMomentum(ele)    
#     pn = np.linalg.norm([px, py, pz])
#     theta_, phi_ = theta(px, py, pz), phi(px, py)
#     print(f'{it} px={px:.1f} py={py:.1f} pz={pz:.1f} pn={pn:.1f} theta={theta_:.3f} phi={phi_:.3f}')
