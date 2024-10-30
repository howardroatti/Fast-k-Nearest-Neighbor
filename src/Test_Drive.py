'''
number of instaces:
iris          -> 150
wine          -> 178
diabetes      -> 442
boston        -> 506
breast cancer -> 569
digits        -> 1797
'''
from model.fkNN import fkNN
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_wine, load_digits
#from joblib import dump, load

import sys
sys.setrecursionlimit(15500)  # Set to a higher value if needed, but use with caution

X_train = [[1, 2], 
        [1, 4], 
        [1, 0],
        [10, 2], 
        [10, 4], 
        [10, 0],
        [33, 2], 
        [33, 4], 
        [33, 0]
        ]
y_train = [0,0,0,1,1,1,2,2,2]

X_test = [[1, 3],
        [10, 1],
        [33, 3]
        ]
y_test = np.array([0,1,2])

data = load_digits()
X = data.data
y = data.target

local_random_state = 42 #Iris Best Result 42

X_train, X_test, y_train, y_test = train_test_split= train_test_split(X, y, test_size=.3, random_state=local_random_state, )

verbose = 0
modelo = fkNN(alpha=0.001, decisionLevel='L0', n_jobs=None, cluster_method='kmeans', n_clusters=3, verbose=verbose, random_state=local_random_state,)
modelo.fit(X_train, y_train)

#dump(modelo, 'fkNN.pkl', compress='zlib')

if verbose == 0:
    print("%f seconds to building tree" % modelo.timeToBuilding)

preds = modelo.predict(X_test)
print(classification_report(y_test, preds))
print("Accuracy: %.3f%%" % (accuracy_score(y_test, preds) * 100))

if verbose == 0:
    print("%f average seconds to categorize" % modelo.timeToClassifying)