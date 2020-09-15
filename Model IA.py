#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Problème de régression linéaire (Apprentissage supervisé)
import numpy as np   
import matplotlib.pyplot as plt   
from sklearn.datasets import make_regression   
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)   
x, y = make_regression(n_samples=100, n_features=1, noise=10)   
plt.scatter(x, y) 

model = SGDRegressor(max_iter=1000, eta0=0.001)  
model.fit(x,y)

print('Coeff R2 =', model.score(x, y))   
plt.scatter(x, y)   
plt.plot(x, model.predict(x), c='red', lw = 3)


# In[7]:


# Problème de régression non-linéaire (Apprentissage supervisé)
import numpy as np   
import matplotlib.pyplot as plt   
from sklearn.datasets import make_regression   
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures


np.random.seed(0)   
   
# création du Dataset   
x, y = make_regression(n_samples=100, n_features=1, noise=10)   
y = y**2 # y ne varie plus linéairement selon x !   
   
   
# On ajoute des variables polynômiales dans notre dataset   
poly_features = PolynomialFeatures(degree=2, include_bias=False)  
x = poly_features.fit_transform(x)   
   
   
plt.scatter(x[:,0], y)   
x.shape # la dimension de x: 100 lignes et 2 colonnes

# On entraine le modele comme avant ! rien ne change ! 
model = SGDRegressor(max_iter=1000, eta0=0.001)   
model.fit(x,y)   
print('Coeff R2 =', model.score(x, y))   
   
plt.scatter(x[:,0], y, marker='o')   
plt.scatter(x[:,0], model.predict(x), c='red', marker='+') 


# In[10]:


# Problème de classification (Apprentissage supervisé)
# Régression Logistique avec Gradient Descent 
import numpy as np   
import matplotlib.pyplot as plt   
from sklearn.datasets import make_classification   
from sklearn.linear_model import SGDClassifier   
# Génération de données aléatoires: 100 exemples, 2 classes, 2 features x0 et x1   
np.random.seed(1)   
X, y = make_classification(n_samples=100,n_features=2, n_redundant=0, n_informative=1,  
                             n_clusters_per_class=1)   
   
# Visualisation des données   
plt.figure(num=None, figsize=(8, 6))   
#plt.scatter(x[:,0], x[:, 1], marker = 'o', c=y, edgecolors='k')   
#plt.xlabel('X0')   
#plt.ylabel('X1')   
#x.shape

# Génération d'un modele en utilisant la fonction cout 'log' pour Logistic Regression   
model = SGDClassifier(max_iter=1000, eta0=0.001, loss='log')   
   
model.fit(X, y)   
print('score:', model.score(x, y))

# Visualisation des données   
h = .02   
colors = "bry"   
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1   
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1   
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),   
                     np.arange(y_min, y_max, h))   
   
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])   
Z = Z.reshape(xx.shape)   
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)   
plt.axis('tight')   
   
for i, color in zip(model.classes_, colors):   
    idx = np.where(y == i)   
    plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired, edgecolor='black', s
=20)


# In[13]:


#K-Nearest Neighbour.(Algorithme de classification)
import numpy as np   
import matplotlib.pyplot as plt   
from sklearn.datasets import load_digits   
from sklearn.neighbors import KNeighborsClassifier

# importons une base de données de chiffre   
digits = load_digits()   
   
X = digits.data   
y = digits.target   
   
print('dimension de X:', X.shape) 

# visualisons un de ces chiffres   
plt.imshow(digits['images'][0], cmap = 'Greys_r')   
   
# Entraînement du modele   
model = KNeighborsClassifier()   
model.fit(X, y)   
model.score(X, y)

#Test du modele   
test = digits['images'][100].reshape(1, -1)   
plt.imshow(digits['images'][100], cmap = 'Greys_r')   
model.predict(test)

