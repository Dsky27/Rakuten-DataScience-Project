import re


# fonction permettant de gérer les problèmes d'encodage, ex en dessous 
def clean_str(string):
            
            string = re.sub(r"Ã©", "é", string)
            string = re.sub(r"Ã¨", "è", string)
            
            return string.strip()


# text="24,Mini Wifi 720p Caméra Drone Rc Quadcopter 24 Ghz 4ch 6-Axis Gyro 3d Ufo Fpv Rc _1975 @Cocoworld-Générique,Mini Wifi 720P Caméra Drone RC Quadcopter 24 GHz 4CH 6-Axis Gyro 3D UFO FPV RC Description: Marque: DHD D4 Point NO.:D4 Fréquence: 24 GHz canal: 4 canaux Axe: 6 axes Couleur: Vert bleu Copter batterie: 3.7V 220mAh temps de vol: A propos 6mins temps de charge: à propos de 50mins contrôle à distance: environ 50 m taille du produit: 105 * 105 * 3cm Caractéristiques: 4 canaux mini quadcopter. le mode de détection de gravité rend plus intéressant de contrôler le drone. Avec la caméra 720P il permet de prendre des photos et enregistrer des vidéos. Wifi FPV en temps réel apporte le monde devant vous. Avec la lumière LED pour la nuit flight.With 3D flips « n » roll peut rendre votre vol plus drôle. vous permettent de voir la vue HD en direct de la caméra FPV et tenir au courant de ce qui se passe dans l&#39;air. la technologie 2.4G adoptée pour l&#39;anti-ingérence. encore plus d&#39;un quadcopter vole en même temps ils ne vont pas interférer les uns avec les autres. 6-Axis vol quad-giravion gyroscope une forte stabilité peut facilement mettre en ¿uvre divers mouvements de vol plus forte résistance au vent plus facile à contrôler. Avec la lumière LED deviennent plus en sécurité dans le ciel nocturne. mini portable cellule de pliage peuvent être montés dans le contrôle à distance de véhicules aériens équipés de plein air sac de vol Voyage facile sans pression Fonctions: haut / bas tourner à gauche / droite avant / arrière gauche / droite sidewayfly flips 3D « n » roll avec LED paquet contient: 1 x Quadcopter (720p camera sans fil) 1 x batterie 1 x télécommande 1 x étui de protection extérieure 1 x support de téléphone 1 x Quadcoper chargeur USB 4 x 4 garde lame x 1 x Manuel peut facilement mettre en ¿uvre divers mouvements de vol plus forte résistance au vent plus facile à contrôler. Avec la lumière LED deviennent plus en sécurité dans le ciel nocturne. mini portable cellule de pliage peuvent être montés dans le contrôle à distance de véhicules aériens équipés de plein air sac de vol Voyage facile sans pression Fonctions: haut / bas tourner à gauche / droite avant / arrière gauche / droite sidewayfly flips 3D « n » roll avec LED paquet contient: 1 x Quadcopter (720p camera sans fil) 1 x batterie 1 x télécommande 1 x étui de protection extérieure 1 x support de téléphone 1 x Quadcoper chargeur USB 4 x 4 garde lame x 1 x Manuel peut facilement mettre en ¿uvre divers mouvements de vol plus forte résistance au vent plus facile à contrôler. Avec la lumière LED deviennent plus en sécurité dans le ciel nocturne. mini portable cellule de pliage peuvent être montés dans le contrôle à distance de véhicules aériens équipés de plein air sac de vol Voyage facile sans pression Fonctions: haut / bas tourner à gauche / droite avant / arrière gauche / droite sidewayfly flips 3D « n » roll avec LED paquet contient: 1 x Quadcopter (720p camera sans fil) 1 x batterie 1 x télécommande 1 x étui de protection extérieure 1 x support de téléphone 1 x Quadcoper chargeur USB 4 x 4 garde lame x 1 x Manuel,3748203527,1240721678"
text="ne pas dÊÄvorer les humains lol "

print(clean_str(text))