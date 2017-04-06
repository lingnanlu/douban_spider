from UBCFBase import UBCFBase

class FS_Weight(UBCFBase):
    def __init__(self, k=40, min_k=1, alpha=0.5,  sim_options={}, **kwargs):
        UBCFBase.__init__(self, k, min_k, sim_options, **kwargs)
        self.alpha = 0.5

    def train(self, trainset):
        UBCFBase.train(self, trainset)

        self.fusion_sim = self.get_fusion_sim(self.sim, self.behavier_sim)

    def get_fusion_sim(self, sim, behavier_sim):
        pass

    def estimate(self, u, i):
        pass



