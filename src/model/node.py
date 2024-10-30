from collections import deque, defaultdict

class Node(object):

    __slots__ = ['tipo', 'index', 'filhos', 'root', 'pLevel', 'L0', 'L1', 'densities', 'odds', 'sameLabels']

    def __init__(self, tipo:str='B'):
        '''
            tipo:
                B - BottomLevel
                H - HyperLevel
                P - P-Level ou Root
            index - indica quem é o nó no conjunto de dados
            filhos - contém a lista de chaves de cada um dos nós que o nó atual possui ligação
        '''
        self.tipo = str(tipo)
        self.index = int(-1)        
        self.filhos = deque()
        if self.tipo == 'P':
            self.root = False
            self.pLevel = int(0)
        if self.tipo == 'H':
            self.L0 = tuple()
            self.L1 = tuple()
            self.densities = defaultdict()
            self.odds = deque()
            self.sameLabels = False
        
    def inserirFilho(self, node:object):
        self.filhos.append(node)

    def removerFilho(self, node:object):
        self.filhos.remove(node)

    def alterarTipo(self, tipo:str):
        self.tipo = tipo

    def getIndex(self)->int:
        return self.index

    def setIndex(self, index:int):
        self.index = index

    def setL0(self, l0:tuple):
        self.L0 = l0

    def getL0(self)->tuple:
        return self.L0

    def getL(self)->int:
        return len(self.filhos)

    def setL1(self, l1:tuple):
        self.L1 = l1

    def getL1(self)->tuple:
        return self.L1

    def setRoot(self):
        self.root = True

    def setLevel(self, level:int):
        self.pLevel = level

    def getLevel(self)->int:
        return self.pLevel

    def inserirFilhos(self, nodes:list=[]):
        for n in nodes:
            self.inserirFilho(n)

    def makeOdds(self):
        total_odds = sum(self.densities.values())
        append = self.odds.append
        for item in self.densities.values():
            append(item/total_odds)

    def setSameLabels(self):
        self.sameLabels = True

    def getSameLabels(self)->bool:
        return self.sameLabels

    def __str__(self)->str:
        if self.tipo == 'B':
            return "[Tipo: {0} | Índice: {1}]".format(self.tipo, self.index)
        elif self.tipo == 'H':
            return "Tipo: {0} \t| Índice: {1} | Filhos\n\tK:{2}\n\n".format(self.tipo, self.index, [filho.__str__() for filho in self.filhos])
        else:
            if self.root:
                return "Tipo: {0} \t| Índice: {1} | Nível: {2}\n\n".format(self.tipo, self.index, self.pLevel)
            else:
                return "Tipo: {0} \t| Índice: {1} | Filhos\n\tK:{2}\n\n".format(self.tipo, self.index, [filho.__str__() for filho in self.filhos])
                