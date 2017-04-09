# coding=utf-8
from UBCFBase import UBCFBase

class FR_Threshold(UBCFBase):

    def __init__(self, k=40, min_k=1, threshold=10, sim_options={}, **kwargs):
        UBCFBase.__init__(self, k, min_k, sim_options, **kwargs)
        self.threshold = threshold

    def train(self, trainset):
        UBCFBase.train(self, trainset)

    def estimate(self, u, i):
        # 通过传统与基于行为的预测结果，进行融合
        pass

    def estimate_by_tr_cf(self, u, i):
        pass

    def estimate_by_behavior_cf(self, u, i):
        pass