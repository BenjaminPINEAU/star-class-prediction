import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Levenshtein


star = pd.read_csv ('https://github.com/YBIFoundation/Dataset/raw/main/Stars.csv')
couleurs_defaut = ['Red', 'Blue White', 'White' ,'Yellowish White' , 'Pale yellow orange' ,'Whitish' , 'Orange', 'Blue']
dataset_defaut = star

#nettoyer les données (les couleurs des étoiles)
def chaine_la_plus_proche(chaine, liste_de_chaines):
  dist=[Levenshtein.distance(chaine, liste_de_chaines[i]) for i in range(len(liste_de_chaines))]
  ind_min = np.argmin(dist)
  return(liste_de_chaines[ind_min])
def nettoyage_couleurs(dataset=dataset_defaut, couleurs = couleurs_defaut):
  data_clean = dataset.copy()
  star_clean['Star color']=star_clean['Star color'].apply(lambda x: chaine_la_plus_proche(x,couleurs ))
  return(star_clean)
