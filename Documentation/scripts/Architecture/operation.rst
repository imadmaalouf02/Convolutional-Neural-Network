Architecture d’un Convolutional Neural Network-CNN
===================================================

-----------------------------------------------------------------------------------



.. figure:: /Documentation/images/arch1.png
   :width:  700
   :align: center
   :alt: Alternative Text



.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i>    
    

    Les CNN désignent une sous-catégorie de réseaux de neurones et sont à ce jour un des modèles de classification d’images réputés être les plus performant.
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    Leur mode de fonctionnement est à première vue simple : l’utilisateur fournit en entrée une image sous la forme d’une matrice de pixels.
    </i></span></p>


Celle-ci dispose de 3 dimensions :


    * **Deux dimensions** pour une image en niveaux de gris.*

    * *Une troisième dimension, de profondeur 3 pour représenter les couleurs fondamentales (Rouge, Vert, Bleu).*



.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    Contrairement à un </span><span style="color:blue;"><strang> modèle MLP (Multi Layers Perceptron)</strang></span><span style="color:#000080;"> classique qui ne contient qu’une partie classification, l’architecture du Convolutional Neural Network dispose en amont d’une partie convolutive et comporte par conséquent deux parties bien distinctes :

   </i></span></p>

.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    -  Une partie convolutive : Son objectif final est d’extraire des caractéristiques propres à chaque image en les compressant de façon à réduire leur taille initiale. En résumé, l’image fournie en entrée passe à travers une succession de filtres, créant par la même occasion de nouvelles images appelées cartes de convolutions. Enfin, les cartes de convolutions obtenues sont concaténées dans un vecteur de caractéristiques appelé code CNN.
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    -  Une partie classification : Le code CNN obtenu en sortie de la partie convolutive est fourni en entrée dans une deuxième partie, constituée de couches entièrement connectées appelées perceptron multicouche (MLP pour Multi Layers Perceptron). Le rôle de cette partie est de combiner les caractéristiques du code CNN afin de classer l’image.
    <p style="text-align: justify;"><span style="color:#000080;"><i> 


.. figure:: /Documentation/images/arch.png
   :width:  700
   :align: center
   :alt: Alternative Text

    *Schéma représentant l’architecture d’un CNN*


.. note::

    Il existe quatre types de couches pour un réseau de neurones convolutif : la couche de **convolution**, la couche de **pooling**, la couche de **correction ReLU** et la couche **fully-connected**. Dans ce chapitre, je vais vous expliquer le fonctionnement de ces différentes couches



--------------------------------------------------------------------------------------------------------------------




