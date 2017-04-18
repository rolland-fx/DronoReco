# DronoReco
PoC for neuronal network  UaV target finder based on Keras 

Le fichier exec-precalc est le seul vraiment interessant


Guide très basique pour que ca fonctionne :

- suivre ca :
http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/
 en utilisant :
	- les versions les plus a jours
	- Python 3.5 (ou 3.6)
	
melanger tout cela avec cela :
 - https://lepisma.github.io/articles/2015/07/30/up-with-theano-and-cuda/?cm_mc_uid=47318398806314855340174&cm_mc_sid_50200000=1492475301

- Les fichiers de configs pour Theanos et Keras devraient avoir juste les paths a modifier
- Bien setup les variables d'environnement avec tout les paths necessaires

les dataset et un modele (pas tres bons, c'est juste la derniere tentative a ce jour) sont disponible ici :

https://drive.google.com/drive/folders/0B9YQLKDfkTOuLWRsS3U4TU4tZDA?usp=sharing

PS : pas besoin de cuda si vous ne voulez pas built de modeles, et alors, l'installation sera bien plus simple

Structure a avoir pour le projet en l'état actuel :
http://imgur.com/a/AHBm3

