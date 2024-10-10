
La couche de convolution
=========================


  
.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>   

    La convolution est une opération mathématique simple généralement utilisée pour le</span><span style="color:red;"> traitement et la reconnaissance d’images. 
    </i></span></p>



À quoi sert la convolution ?
----------------------------




.. figure:: /Documentation/images/conv1.png
   :width: 100%
   :alt: Alternative text for the image
   :name: logo


  
.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>     
    <Strang>La couche de convolution</Strang></span><span style="color:#000080;"> est la composante clé des réseaux de neurones convolutifs, et constitue toujours au moins leur première couche.
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i>   
    Son but est de repérer la présence d'un ensemble de features dans les images reçues en entrée. 
     </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i>      
    Pour cela, on réalise un </span><span style="color:blue;">filtrage par convolution </span><span style="color:#000080;">: le principe est de faire "glisser" une fenêtre représentant la feature sur l'image, et de calculer le produit de convolution entre la feature et chaque portion de l'image balayée. 
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i>       
    Une feature est alors vue comme un filtre : les deux termes sont équivalents dans ce contexte. 
    </i></span></p>


.. admonition::  Remarque

   .. container:: blue-box

    Cette technique est très proche de celle étudiée dans la partie précédente pour faire du template matching : ici, c'est le produit convolution qui est calculé, et non la corrélation croisée.


    .. figure:: /Documentation/images/Exmple.png
        :width: 100%
        :alt: Alternative text for the image
        :name: logo




.. figure:: /Documentation/images/conv.png
   :width: 100%
   :alt: Alternative text for the image
   :name: logo



.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i>   
    Dans un premier temps, on définit la taille de la </span><span style="color:blue;">fenêtre de filtre</span><span style="color:#000080;"> située en haut à gauche.
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i>     
    La </span><span style="color:blue;">fenêtre de filtre</span><span style="color:#000080;">, représentant la feature, se déplace progressivement de la gauche vers la droite d’un certain nombre de cases défini au préalable (le pas) jusqu’à arriver au bout de l’image.
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i>     
    À chaque portion d’image rencontrée, un calcul de convolution s’effectue permettant d’obtenir en sortie une carte d’activation ou feature map qui indique où est localisées les features dans l’image : plus la feature map est élevée, plus la portion de l’image balayée ressemble à la feature.
    </i></span></p>



Exemple d’un filtre de convolution classique
-----------------------------------------


.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i>  
    Lors de la partie convolutive d’un Convolutional Neural Network, l’image fournie en entrée passe à travers une </span><span style="color:blue;">succession de filtres de convolution.</span><span style="color:#000080;"> Par exemple, il existe des filtres de convolution fréquemment utilisés et permettant d’extraire des caractéristiques plus pertinentes que des pixels comme la détection des bords </span><span style="color:blue;">(filtre dérivateur)</span><span style="color:#000080;"> ou des formes géométriques. Le choix et l’application des filtres se fait </span><span style="color:blue;">automatiquement</span><span style="color:#000080;"> par le modèle.
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    Parmi les filtres les plus connus, on retrouve notamment le </span><span style="color:blue;">filtre moyenneur</span><span style="color:#000080;"> (calcule pour chaque pixel la moyenne du pixel avec ses 8 proches voisins) ou encore le </span><span style="color:blue;">filtre gaussien</span><span style="color:#000080;"> permettant de réduire le bruit d’une image fournie en entrée :
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    Voici un exemple des effets de ces deux différents filtres sur une image comportant un bruit important (on peut penser à une photographie prise avec une faible luminosité par exemple). Toutefois, un des inconvénients de la réduction du bruit est qu’elle s’accompagne généralement d’une réduction de la netteté :

    </i></span></p>




.. figure:: /Documentation/images/Exmple1.png
   :width: 100%
   :alt: Alternative text for the image
   :name: logo



.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i>  

    Comme on peut l’observer, contrairement au filtre moyenneur, le filtre gaussien réduit le bruit sans pour autant réduire significativement la netteté. 
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    Outre sa fonction de filtrage, l’intérêt de </span><span style="color:blue;">la partie convolutive d’un CNN</span><span style="color:#000080;">  est qu’elle permet </span><span style="color:blue;">d’extraire des caractéristiques propres à chaque image</span><span style="color:#000080;">  en les compressant de façon à réduire leur taille initiale, via des méthodes de sous-échantillonnage tel que le Max-Pooling.
    </i></span></p>
















