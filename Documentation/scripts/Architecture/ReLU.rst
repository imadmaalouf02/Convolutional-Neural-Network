La couche de correction ReLU
=============================

Introduction
--------------


.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i> 

    Pour améliorer l'efficacité du traitement en intercalant entre les couches de traitement une couche qui va opérer une fonction mathématique (fonction d'activation) sur les signaux de sortie. dans ce cadre on trouve
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    ReLU (Rectified Linear Units) désigne la fonction réelle non-linéaire
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    définie par ReLU(x)=max(0,x).
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    La couche de correction ReLU remplace donc toutes les valeurs négatives reçues en entrées par des zéros. Elle joue le rôle de fonction d'activation
    </i></span></p> 


Normalization
--------------
Changer tout négatif à zéro


.. figure:: /Documentation/images/R1.png
   :width: 100%
   :alt: Alternative text for the image
   :name: logo



.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i> 

    Un élément important dans l’ensemble du processus est l’Unité linéaire rectifiée ou ReLU. Les mathématiques derrière ce concept sont assez simples encore une fois: chaque fois qu’il y a une valeur négative dans un pixel, on la remplace par un 0. Ainsi, on permet au CNN de rester en bonne santé (mathématiquement parlant) en empêchant les valeurs apprises de rester coincer autour de 0 ou d’exploser vers l’infinie.
    </i></span></p> 



.. figure:: /Documentation/images/R2.png
   :width: 100%
   :alt: Alternative text for the image
   :name: logo



.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    C’est un outil pas vraiment sexy mais fondamental car sans lequel le CNN ne produirait pas vraiment les résultats qu’on lui connaît.
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i> 

    Le résultat d’une couche ReLU est de la même taille que ce qui lui est passé en entrée, avec simplement toutes les valeurs négatives éliminées.
     </i></span></p>

    <p style="text-align: justify;"><span style="color:blue;"><i>    
    La sortie de l'un devient l'entrée du suivant.
    </i></span></p>

.. figure:: /Documentation/images/R3.png
   :width: 100%
   :alt: Alternative text for the image
   :name: logo





