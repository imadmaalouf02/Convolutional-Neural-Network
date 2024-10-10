La couche de pooling
======================



.. figure:: /Documentation/images/1.webp
   :width: 100%
   :alt: Alternative text for the image
   :name: logo



.. figure:: /Documentation/images/2.webp
   :width: 100%
   :alt: Alternative text for the image
   :name: logo



.. admonition::  But

   .. container:: blue-box

    Le but du Pooling est de réduire la taille des “images” mais également de pallier le phénomène d’ « overfitting ». Tout comme pour la convolution on applique un filtre qu’on fait glisser sur l’image et (dans la version “max pooling”) on garde la valeur max sur chaque fenêtre. On parle aussi de « Down-Sampling » ou « Sub-Sampling »




À quoi sert la pooling?
-------------------------
  

.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    
    Ce type de couche est souvent placé entre deux couches de convolution : elle reçoit en entrée plusieurs feature maps, et applique à chacune d'entre elles l'opération de pooling. 
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    
    L'opération de pooling ( ou sub-sampling)consiste à réduire la taille des images, tout en préservant leurs caractéristiques importantes.

    </i></span></p>


    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    
    Pour cela, on découpe l'image en cellules régulière, puis on garde au sein de chaque cellule la valeur maximale. En pratique, on utilise souvent des cellules carrées de petite taille pour ne pas perdre trop d'informations. Les choix les plus communs sont des cellules adjacentes de taille 2 × 2 pixels qui ne se chevauchent pas, ou des cellules de taille 3 × 3 pixels, distantes les unes des autres d'un pas de 2 pixels (qui se chevauchent donc). On obtient en sortie le même nombre de feature maps qu'en entrée, mais celles-ci sont bien plus petites.
    </i></span></p>


    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    La couche de pooling permet de réduire le nombre de paramètres et de calculs dans le réseau. On améliore ainsi l'efficacité du réseau et on évite le sur-apprentissage.
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    Il est courant d'insérer périodiquement une couche Pooling entre les couches Conv successives dans une architecture ConvNet. Sa fonction est de réduire progressivement la taille spatiale de la représentation afin de réduire la quantité de paramètres et de calculs dans le réseau, et donc de contrôler également le sur-ajustement. La couche de pooling fonctionne indépendamment sur chaque tranche de profondeur de l'entrée et la redimensionne spatialement, en utilisant l'opération MAX.
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    Ainsi, la couche de pooling rend le réseau moins sensible à la position des features : le fait qu'une feature se situe un peu plus en haut ou en bas, ou même qu'elle ait une orientation légèrement différente ne devrait pas provoquer un changement radical dans la classification de l'image.

    </i></span></p>



.. note::

    Pooling: réduire la pile d'images



---------------------------------------------------------------------------------------------------------

Exmple
---------


**les etapes de pooling**



.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    - Choisissez une taille de fenêtre (généralement 2 ou 3).
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    - Choisissez un pas (généralement 2).
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    - Parcourez votre fenêtre à travers vos images filtrées.
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    - De chaque fenêtre, prenez la valeur maximale.
    </i></span></p>



.. figure:: /Documentation/images/E1.png
   :width: 100%
   :alt: Alternative text for the image
   :name: logo



.. figure:: /Documentation/images/E2.png
   :width: 100%
   :alt: Alternative text for the image
   :name: logo




.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    Après avoir procédé au pooling, l’image n’a plus qu’un quart du nombre de ses pixels de départ.
    </i></span></p>


.. figure:: /Documentation/images/E3.png
   :width: 100%
   :alt: Alternative text for the image
   :name: logo




.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    Parce qu’il garde à chaque pas la valeur maximale contenue dans la fenêtre, il préserve les meilleurs caractéristiques de cette fenêtre. Cela signifie qu’il ne se préoccupe pas vraiment d’où a été extraite la caractéristique dans la fenêtre.
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    Le résultat est que le CNN peut trouver si une caractéristique est dans une image, sans se soucier de l’endroit où elle se trouve. Cela aide notamment à résoudre le problème liés au fait que les ordinateurs soient hyper-littéraires.
    </i></span></p>



.. figure:: /Documentation/images/E4.png
   :width: 100%
   :alt: Alternative text for the image
   :name: logo


.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    Au final, une couche de pooling est simplement un traitement de pooling sur une image ou une collection d’images. L’output aura le même nombre d’images mais chaque images aura un nombre inférieur de pixels. Cela permettra ainsi de diminuer la charge de calculs. Par exemple, en transformant une image de 8 mégapixels en une image de 2 mégapixels, ce qui nous rendra la vie beaucoup plus facile pour le reste des opérations à effectuer par la suite.
    </i></span></p>






