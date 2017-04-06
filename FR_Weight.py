from UBCFBase import UBCFBase

class FR_Weight(UBCFBase):
    def __init__(self, k=40, min_k=1, alpha=0.5,  sim_options={}, **kwargs):
        UBCFBase.__init__(self, k, min_k, sim_options, **kwargs)
        self.alpha = alpha

    def train(self, trainset):
        UBCFBase.train(self, trainset)

    def estimate(self, u, i):
        pass

    def estimate_by_tr_cf(self, u, i):
        pass

    def estimate_by_behavior_cf(self, u, i):
        pass