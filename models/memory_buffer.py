import torch

class PPOMemory:
    def __init__(self, bs):
        self.s = []
        self.a = []
        self.v = []
        self.p = []
        self.r = []
        self.t = []

        self.bs = bs

    def load(self, mems):
        for mem in mems:
            self.s += mem.s; self.a += mem.a
            self.v += mem.v; self.p += mem.p
            self.r += mem.r; self.t += mem.t

        return self

    def remember(self, s,a,v,p,r,t):
        self.s.append(s)
        self.a.append(a)
        self.v.append(v)
        self.p.append(p)
        self.r.append(r)
        self.t.append(t)

    def clear(self):
        self.s = []; self.a = []
        self.v = []; self.p = []
        self.r = []; self.t = []

    def get_batches(self):
        idxs = torch.randperm(len(self.a))
        batch_idxs = idxs.split(self.bs)

        return self.s, self.a, self.v, \
            self.p, self.r, self.t, batch_idxs

    def combine(self, others):
        for o in others:
            self.s += o.s; self.a += o.a
            self.v += o.v; self.p += o.p
            self.r += o.r; self.t += o.t

        return self