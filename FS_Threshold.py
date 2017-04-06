# coding=utf-8
from UBCFBase import UBCFBase

class FS_Threshold(UBCFBase):

    def __init__(self, k=40, min_k=1, threshold=10, sim_options={}, **kwargs):
        UBCFBase.__init__(self, k, min_k, sim_options, **kwargs)
        self.threshold = threshold

    def train(self, trainset):

        UBCFBase.train(self, trainset)

        self.fusion_sim = self.get_fusion_sim(self.sim, self.behavier_sim)

    def estimate(self, u, i):
        # 利用融合相似度进行计算
        pass

    def get_fusion_sim(self, sim, behavier_sim):
        pass


