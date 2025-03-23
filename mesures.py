import numpy as np
from sklearn.datasets import make_classification
from scipy.stats import chi2, norm
import matplotlib.pyplot as plt
from math import atan, exp, pi, log

A=0.05
B=0.6
def f(x_1,x_2, a=A, b=B):
  '''effectue une transformation affine en échelle log (de l'argument maximal) tq :
  f(a) = arctan(-5) (1ère l'inflexion)
  f(b) = arctan(5) (2ème l'inflexion)
  '''
  x = np.max([x_1, x_2])
  cd= 10/(log(b)-log(a))
  oo= -5 - log(a)*cd
  if x==0:
    return 0

  return(atan(log(x)*cd + oo)/pi+0.5)



def covariance_empirique(X):
  """
  Calculates the empirical covariance matrix of a dataset.

  Args:
    X: A NumPy array representing the dataset, where each row is a sample
       and each column is a feature.

  Returns:
    A NumPy array representing the empirical covariance matrix.
  """
  n = X.shape[0]  # Number of samples
  d = X.shape[1]  # Number of features
  # Center the data (subtract the mean of each feature)
  X_centered = X - np.mean(X, axis=0)
  # Calculate the covariance matrix
  covariance_matrix = np.dot(X_centered.T, X_centered) / (n - 1)
  return covariance_matrix




def mesure_quantiles(X, y, alpha=0.99999, mu_a=0, Sigma_a=0, mu_b=0, Sigma_b=0, parametres_forcees = False, combinaison  = f) : #on se donne un quantile, on regarde quelle proportion d'éléments de la classe 2 sont dans la classe 1, vice versa et on en dédeuit un truc
  #print('feurfeurfeurfeurfeur')
  """
  Donne une valeur de distance (entre 0 & 1) entre deux gaussiennes.
  Considère les ellipses définies par un quantile, et plus exactement la proportion des points de l'autre gaussienne qui sont dans l'ellipse.
  le résultat final est une fonction de ces deux proportions

  Args:
   X : coordonnées des points dans l'espace (réels)
   Y : vecteur d'appartenance à une classe (tout types, toutes valeurs tant qu'il y en a exactement 1 unique type et deux différentes valeurs)
   alpha : quantile choisi (entre 0 et 1) (par défaut, 0.99999)
   mu_a : moyenne de la gaussienne A (par défaut, moyenne empirique)
   Sigma_a : covariance de la gaussienne A (par défaut, matrice de covariance empirique)
   mu_b : moyenne de la gaussienne B (par défaut, moyenne empirique)
   Sigma_b : covariance de la gaussienne B (par défaut, matrice de covariance empirique
   paramètres_forcés : si les moyennes / variances sont connues, mettre ce paramètre à True, et choisir des valeurs pour mu_a, mu_b, Sigma_a, Sig (booléen)(par défaut: False)
   combinaison : prends deux proportions (des points dans les ellipses), et renvoie un score. Par défaut : f

  """
  a,b = np.unique(y)[0], np.unique(y)[1]
  points_classe_a = X[y == a]
  points_classe_b = X[y == b]
  if parametres_forcees :
    m_a,m_b = mu_a, mu_b
    S_a,S_b = Sigma_a, Sigma_b

  else :#si l'utilisateur n'a pas entré de valeurs pour mu_a etc, on les calcule à la main
    m_a,m_b = np.mean(points_classe_a, axis= 0), np.mean(points_classe_b, axis= 0)
    S_a = covariance_empirique(points_classe_a)
    S_b = covariance_empirique(points_classe_b)

  dimension = len(m_a)  # Dimension de la gaussienne
  quantile = np.sqrt(chi2.ppf(alpha, df=dimension))

  #on compte la proportion de points de A dans le quantile alpha de B
  cpt_a =0
  S_inv_b= np.linalg.inv(S_b)
  for x_a in points_classe_a:
    delta_x_a = x_a - m_b
    dist_mahalanobis_b = np.sqrt(delta_x_a.T @ S_inv_b@ delta_x_a)
    if dist_mahalanobis_b < quantile:
      est_dans_quantile = True
    else:
      est_dans_quantile = False
    if est_dans_quantile:
      cpt_a += 1
  prop_a = cpt_a / len(points_classe_a)
  print('feur_a', prop_a)

  #idem pour b
  cpt_b =0

  S_inv_a= np.linalg.inv(S_a)
  for x_b in points_classe_b:
    delta_x_b = x_b - m_a
    dist_mahalanobis_a = np.sqrt(delta_x_b.T @ S_inv_b @ delta_x_b)
    if dist_mahalanobis_a < quantile:
      est_dans_quantile = True
    else:
      est_dans_quantile = False
    if est_dans_quantile:
      cpt_b += 1
  prop_b = cpt_b/ len(points_classe_b)
  print('feur_b', prop_b)
  return(f(prop_a, prop_b))