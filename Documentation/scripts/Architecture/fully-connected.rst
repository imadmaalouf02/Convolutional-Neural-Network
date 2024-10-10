La couche fully-connected
==========================

.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    
    Les CNNs ont une autre flèche dans leur carquois. En effet, les couches entièrement connectés prennent les images filtrées de haut niveau et les traduisent en votes. Dans notre exemple, nous devons seulement décider entre deux catégories, X et O.




.. figure:: /Documentation/images/F3.PNG
   :width: 700
   :align: center
   :alt: Alternative text for the image




.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    
    Les couches entièrement connectées sont les principaux blocs de construction des réseaux de neurones traditionnels. Au lieu de traiter les inputs comme des tableaux de 2 Dimensions, ils sont traités en tant que liste unique et tous traités de manière identique. Chaque valeur a son propre vote quant à si l’image est un X ou un O. Cependant, le process n’est pas complètement démocratique. Certaines valeurs sont bien meilleures à détecter lorsqu’une i mage est un X que d’autres, et d’autres sont bien meilleures à détecter un O. Celles-ci ont donc davantage de pouvoir de vote que les autres. Ce vote est appelé le poids, ou la force de la connection, entre chaque valeur et chaque catégorie.
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    
    Lorsqu’une nouvelle image est présentée au CNN, elle se répand à travers les couches inférieures jusqu’à atteindre la couche finale entièrement connectée. L’élection a ensuite lieu. Et la solution avec le plus de vote gagne et est déclarée la catégorie de l’image.
    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i> 
    
    Les couches entièrement connectées, tout comme les autres couches, peuvent être ajoutées les unes à la suite des autres car leur valeur en sortie (une liste de votes) ressemble énormément à leur valeur en entrée (une liste de valeur). En pratique, plusieurs couches entièrement connectées sont souvent ajoutées les unes à la suite des autres, avec chaque couche intermédiaire votant pour des catégories “cachées” fantômes. En effet, chaque couche additionnelle laisse le réseau apprendre des combinaisons chaque fois plus complexes de caractéristiques qui aident à améliorer la prise de décision.

    </i></span></p>



.. figure:: /Documentation/images/F4.png
   :width: 700
   :align: center
   :alt: Alternative text for the image





