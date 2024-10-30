# https://scikit-learn.org/stable/developers/develop.html
# https://sklearn-template.readthedocs.io/en/latest/user_guide.html
# https://gist.github.com/celsoernani/28d7b1cb00c16142dab43aa654cbd714
# https://wiki.python.org/moin/PythonSpeed/PerformanceTips
# https://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat
# http://docs.python.org/reference/datamodel.html#slots

from model.node import Node
from model.externalCluster import ExternalCluster

import time
from itertools import product
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
#from scipy.spatial.distance import cosine
#from sklearn.metrics import pairwise_distances

class fkNN(BaseEstimator, ClassifierMixin):

    __slots__ = ['alpha', 'L', 'k',  
                 'hyperlevel_size', '__hyperlevel_list', '__root', 
                 'decisionLevel', 'timeToBuilding', 'n_jobs', 
                 'cluster_method', 'n_clusters', 'verbose', 'timeToClassifying',
                 'random_state', 'file_path', 'encoding']

    def __init__(self, alpha:float=0.001, L:int=5, k:int=5, decisionLevel:str='L0', n_jobs:int=None, cluster_method:str='kmeans', 
                 n_clusters:int=2, verbose:int=0, random_state:int=None, file_path:str=None, encoding:str=None):
        self.alpha = alpha
        self.L = L
        self.k = k
        self.hyperlevel_size = 0
        self.__hyperlevel_list = []
        self.__root = None
        self.decisionLevel = decisionLevel
        self.timeToBuilding = float(0.0)
        self.timeToClassifying = float(0.0)
        self.n_jobs = n_jobs
        self.cluster_method = cluster_method
        self.n_clusters = n_clusters
        self.verbose = verbose
        self.random_state = random_state

        if self.cluster_method == 'external':
            self.file_path = file_path
            self.encoding = encoding

    def calculate_cosine_similarity(self, a:np.array, b:np.array)->float:
        # https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
        # https://en.wikipedia.org/wiki/Cosine_similarity
        cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        #cosine_sim = 1 - float(cosine(a, b))

        #dist = pairwise_distances(self.data[c].reshape(1, -1), centroid, metric="cosine")[0, 0]
        return cosine_sim

    def __dissimilarities(self, level:int=0, indexes_list:list=[])->tuple:
        indexes = (-1,-1)
        filter = np.array([(i, j) for i, j in product(indexes_list, repeat=2) if (i > j)])
        
        if level == 0:
            minimum_value_l0 = np.Infinity
            for i, j in filter:
                sim = self.calculate_cosine_similarity(self.X_[i], self.X_[j])
                if sim < minimum_value_l0:
                    minimum_value_l0 = sim
                    indexes = (i,j)

        elif level == 1:
            minimum_value_l0 = np.Infinity
            minimum_value_l1 = np.Infinity
            indexes_l0 = (-1,-1)
            indexes_l1 = (-1,-1)
            
            for i, j in filter:
                sim = self.calculate_cosine_similarity(self.X_[i], self.X_[j])
                if sim < minimum_value_l0:
                    minimum_value_l1 = minimum_value_l0
                    minimum_value_l0 = sim
                    indexes_l1 = indexes_l0
                    indexes_l0 = (i,j)
                elif (sim < minimum_value_l1 and sim != minimum_value_l0):
                    minimum_value_l1 = sim
                    indexes_l1 = (i,j)

            indexes = indexes_l1
        
        if self.verbose == 1:
            print("Level:", level, "Index:", indexes)
            print()
        elif self.verbose == 2:
            print("Level:", level, "Index:", indexes)
            print("Dissimilaritie:", sim)            
            print()
        
        return indexes
    
    def __clustering(self, X):
        if self.cluster_method == 'kmeans':
            return KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(X)
        elif self.cluster_method == 'agglomerative':
            return AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward').fit(X)
        elif self.cluster_method == 'external':
            myCluster = ExternalCluster()
            with open(self.file_path, 'r',encoding=self.encoding) as f:
                f.seek(0)
                for line in f:
                    myCluster.labels_.append(int(line))

                myCluster.n_clusters = len(np.unique(myCluster.labels_))
        else:
            return None

    def showHyperlevel(self):
        for i in range(0, self.hyperlevel_size):
            print(self.__hyperlevel_list[i].__str__())
            print("Densities:", self.__hyperlevel_list[i].densities)
            print("Odds:", self.__hyperlevel_list[i].odds)
            print()
    
    def __supportEta(self, indexes_list:list=[])->tuple:
        node_list = np.array([self.calculate_cosine_similarity(self.X_[i], self.X_[j]) for i, j in product(indexes_list, repeat=2) if (i > j)])
        
        # Calcula a Média das Similaridades        
        average = np.mean(node_list)
        
        # Calcula o Desvio Padrão das Similaridades
        std = np.std(node_list)

        return (average, std)

    def __computeEta(self, pCandidates:list=[], pLevelCount:int=1)->float:
        indexes_list = []
        append = indexes_list.append
        for p in pCandidates:
            append(p.getIndex())
        
        average, std = self.__supportEta(indexes_list=indexes_list)
        eta = average - (self.alpha*(std/1+pLevelCount))

        return eta

    def __createHyperlevel(self, clusters:object=None, cluster_number:int=-1):
        hypernode = Node('H')
        indice = -1
        indexes_list = []
        hypernode.densities = {cl: 0 for cl in self.classes_}

        if self.verbose != 0:
            print("Cluster H:", cluster_number)
        
        append = indexes_list.append
        for j in clusters.labels_:
            indice += 1
            if cluster_number == j:
                bottomnode = Node()
                bottomnode.setIndex(indice)

                hypernode.inserirFilho(bottomnode)
                hypernode.densities[self.y_[indice]] += 1

                append(indice)
            else:
                continue
        
        if self.decisionLevel == 'L0':
            L0 = self.__dissimilarities(level=0, indexes_list=indexes_list)
            hypernode.setL0(L0)
            if self.y_[L0[0]] == self.y_[L0[1]]:
                hypernode.setSameLabels()
        else:
            L1 = self.__dissimilarities(level=1, indexes_list=indexes_list)
            hypernode.setL1(L1)
            if self.y_[L1[0]] == self.y_[L1[1]]:
                hypernode.setSameLabels()
        
        hypernode.makeOdds()
        self.__hyperlevel_list.append(hypernode)

    def __sorter(self, item:object)->int:
        return item.getL()

    def __createPLevels(self, pCandidates:list=[], pLevelCount:int=0)->object:
        if len(pCandidates) == 1:
            root = Node('P')
            root.setIndex(pCandidates[0].getIndex())
            root.inserirFilho(pCandidates[0])
            root.setRoot()
            root.setLevel(pLevelCount)
            return root
        elif len(pCandidates) == 2:
            root = Node('P')
            root.setIndex(pCandidates[0].getIndex())
            root.inserirFilhos(pCandidates)
            root.setRoot()
            root.setLevel(pLevelCount)
            return root
        else:
            eta = self.__computeEta(pCandidates, pLevelCount)
            temp_pCandidates = pCandidates.copy()
            pList = []
            
            append = pList.append
            while len(temp_pCandidates) > 0:
                pNode = Node('P')
                pNode.setIndex(temp_pCandidates[0].getIndex())
                pNode.inserirFilho(temp_pCandidates[0])
                pNode.setLevel(pLevelCount)
                append(pNode)

                temp_pCandidates.remove(temp_pCandidates[0])
                for p in temp_pCandidates:
                    sim = self.calculate_cosine_similarity(self.X_[pNode.getIndex()], self.X_[p.getIndex()])
                    
                    if self.verbose != 0:
                        print("Level: %d\teta: %f\tsim: %f\tIndexes: (%d, %d)" % (pLevelCount, eta, sim, pNode.getIndex(), p.getIndex()))

                    if sim >= eta:
                        pNode.inserirFilho(p)
                        temp_pCandidates.remove(p)
            
            contador = pLevelCount + 1
            root = self.__createPLevels(pList, contador)
            del pList
            return root
    
    def __buildTree(self, X):
        np.random.seed(self.random_state)
        start_time = time.perf_counter ()
         
        # Aplly clustering algorithm
        clusters = self.__clustering(X)
        if clusters == None:
            print("Clustering Method Unknown\nThe options currently are: kmeans and agglomerative")
            return
        # Recover total of clusters to support creation of HyperLevel
        self.hyperlevel_size = clusters.n_clusters
        
        # Creation of Hyperlevel, level to decision make immediately above BottomLevel
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            for i in range(0, self.hyperlevel_size):
                executor.submit(self.__createHyperlevel, clusters, i)
        #Sequencial
        # for i in range(0, self.hyperlevel_size):
        #    self.__createHyperlevel(clusters, i)
   
        # Creation of pLevels
        pCandidate = []
        append = pCandidate.append
        # Randomly searches HyperLevel representatives to be pLevel candidates
        self.__hyperlevel_list = sorted(self.__hyperlevel_list, key=self.__sorter, reverse=True)
        for i in self.__hyperlevel_list:
            if self.decisionLevel == 'L0':
                hyper_idx = i.getL0()[np.random.randint(0,2)]
                i.setIndex(hyper_idx)
                append(i)
            else:
                hyper_idx = i.getL1()[np.random.randint(0,2)]
                i.setIndex(hyper_idx)
                append(i)

        self.__root = self.__createPLevels(pCandidates=pCandidate, pLevelCount=1)

        end_time = time.perf_counter()
        self.timeToBuilding = end_time - start_time

        # Free memory usage
        del pCandidate
        del clusters

        if self.verbose != 0:
            print(self.__root)
            self.showHyperlevel()
            print("%f seconds to building tree" % self.timeToBuilding)

    def last(self, n:int)->int:
        return n[len(n)-1]

    def __searchTree(self, x):
        # https://www.geeksforgeeks.org/python-sort-tuples-increasing-order-key/
        if self.__root == None:
            return
        
        level = self.__root.getLevel()
        initialListOfNodes = [(_, self.calculate_cosine_similarity(x, self.X_[_.getIndex()])) for _ in self.__root.filhos]
        initialListOfNodes = sorted(initialListOfNodes, key = self.last, reverse=True)
        if len(initialListOfNodes) >= self.L:
            initialListOfNodes = initialListOfNodes[:self.L]

        if level - 1 > 0:
            while level - 1 > 0:
                # https://spapas.github.io/2016/04/27/python-nested-list-comprehensions/
                hyperlevelListOfNodes = [(_, self.calculate_cosine_similarity(x, self.X_[_.getIndex()])) for __ in initialListOfNodes for _ in __[0].filhos]
                hyperlevelListOfNodes = sorted(hyperlevelListOfNodes, key = self.last, reverse=True)
                if len(hyperlevelListOfNodes) >= self.L:
                    initialListOfNodes = hyperlevelListOfNodes[:self.L]
                else:
                    initialListOfNodes = hyperlevelListOfNodes
                level -= 1
            # At the end initialListOfNodes will have nodes of HyperLevel            
            hyperlevelListOfNodes = initialListOfNodes
            del initialListOfNodes
        else:
            hyperlevelListOfNodes = initialListOfNodes
            del initialListOfNodes

        # HyperLevel Classification - Start
        hyperlevelListOfNodes_median = []
        for h, sim in hyperlevelListOfNodes:            
            if self.decisionLevel == 'L0':
                temporaryLn = h.getL0()
            elif self.decisionLevel == 'L1':
                temporaryLn = h.getL1()

            sim1 = self.calculate_cosine_similarity(x, self.X_[temporaryLn[0]])
            sim2 = self.calculate_cosine_similarity(x, self.X_[temporaryLn[1]])
            hyperlevelListOfNodes_median.append((h, np.mean([sim1, sim2])))
        
        del hyperlevelListOfNodes

        hyperlevelListOfNodes_median = sorted(hyperlevelListOfNodes_median, key = self.last, reverse=True)

        if len(hyperlevelListOfNodes_median) >= self.L:
            hyperlevelListOfNodes_median = hyperlevelListOfNodes_median[:self.L]

        if hyperlevelListOfNodes_median[0][0].getSameLabels():
            #print("H")
            if self.decisionLevel == 'L0':
                return self.y_[hyperlevelListOfNodes_median[0][0].getL0()[0]]
            elif self.decisionLevel == 'L1':
                return self.y_[hyperlevelListOfNodes_median[0][0].getL1()[0]]  
        # HyperLevel Classification - End
        else:
        # BottomLevel Classification - Start
            #print("B")            
            l = [_.getIndex() for _ in hyperlevelListOfNodes_median[0][0].filhos]
            local_k = self.k if self.k <= len(l) else len(l)
            knn = KNeighborsClassifier(n_neighbors=local_k, metric='cosine')
            knn.fit(self.X_[l] , self.y_[l])
            return knn.predict(np.array(x).reshape(1,-1))[0]
        # BottomLevel Classification - end

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        self.__buildTree(X)

        self._isfitted = True

        return self

    def predict(self, X):
        check_is_fitted(self, 'classes_')

        X = check_array(X)

        averageTime = float(0.0)

        categorized = []

        if len(X) > 1:            
            for x in X:
                start_time = time.perf_counter ()

                categorized.append(self.__searchTree(x))

                end_time = time.perf_counter()
                averageTime += end_time - start_time

            if self.verbose != 0:
                print("Total time to classifying:", averageTime)
            self.timeToClassifying = averageTime / len(X)
        else:
            start_time = time.perf_counter ()

            categorized.append(self.__searchTree(X))
            
            end_time = time.perf_counter()
            averageTime += end_time - start_time
        
        return categorized

    def predict_proba(self, X):
        # Implement softmax to recover probabilities
        print("Not implemented yet.")
        pass