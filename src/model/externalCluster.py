from collections import deque


class ExternalCluster(object):
    __slots__ = ['n_clusters', 'labels_']

    def __init__(self):
        self.n_clusters = int(-1)
        self.labels_ = deque()