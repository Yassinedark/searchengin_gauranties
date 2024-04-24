import streamlit as st
import pandas as pd 
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np
from scipy.spatial.distance import cdist





jp='{"Unnamed: 0":{"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":10,"9":13,"10":15,"11":16,"12":17,"13":19,"14":22,"15":23,"16":24,"17":25,"18":26,"19":28,"20":29,"21":30,"22":32,"23":33,"24":34,"25":35,"26":36,"27":37,"28":38,"29":39,"30":40,"31":41,"32":42,"33":43,"34":44,"35":45,"36":46,"37":47,"38":48,"39":50,"40":51,"41":52,"42":53,"43":54,"44":55,"45":58,"46":59,"47":60,"48":61,"49":62,"50":63,"51":64,"52":65,"53":66,"54":67,"55":68,"56":69},"Type Garanties":{"0":"HOSPITALISATION","1":"HOSPITALISATION","2":"HOSPITALISATION","3":"HOSPITALISATION","4":"HOSPITALISATION","5":"HOSPITALISATION","6":"HOSPITALISATION","7":"HOSPITALISATION","8":"DENTAIRE","9":"DENTAIRE","10":"DENTAIRE","11":"DENTAIRE","12":"DENTAIRE","13":"DENTAIRE","14":"DENTAIRE","15":"DENTAIRE","16":"DENTAIRE","17":"DENTAIRE","18":"DENTAIRE","19":"DENTAIRE","20":"OPTIQUE","21":"OPTIQUE","22":"OPTIQUE","23":"OPTIQUE","24":"OPTIQUE","25":"SOINS COURANTS","26":"SOINS COURANTS","27":"SOINS COURANTS","28":"SOINS COURANTS","29":"SOINS COURANTS","30":"SOINS COURANTS","31":"SOINS COURANTS","32":"SOINS COURANTS","33":"SOINS COURANTS","34":"SOINS COURANTS","35":"SOINS COURANTS","36":"SOINS COURANTS","37":"SOINS COURANTS","38":"SOINS COURANTS","39":"SOINS COURANTS","40":"SOINS COURANTS","41":"SOINS COURANTS","42":"AIDES AUDITIVES","43":"AIDES AUDITIVES","44":"AIDES AUDITIVES","45":"AIDES AUDITIVES","46":"AUTRES POSTES","47":"AUTRES POSTES","48":"AUTRES POSTES","49":"AUTRES POSTES","50":"AUTRES POSTES","51":"AUTRES POSTES","52":"AUTRES POSTES","53":"AUTRES POSTES","54":"AUTRES POSTES","55":"AUTRES POSTES","56":"AUTRES POSTES"},"Garantie 1":{"0":"Frais de s\\u00e9jour","1":"Frais de s\\u00e9jour","2":"Honoraires","3":"Honoraires","4":"Forfait journalier hospitalier","5":"Forfait actes lourds","6":"Chambre particuli\\u00e8re ( y compris maternit\\u00e9 )","7":"frais de long s\\u00e9jour","8":"Soins ( Hors 100 % Sant\\u00e9 )","9":"Proth\\u00e8ses ( Hors 100 % Sant\\u00e9 )","10":"Autres","11":"Autres","12":"Autres","13":"Autres","14":"Autres","15":"Autres","16":"Autres","17":"Autres","18":"Orthodontie","19":"Implantologie dentaire Forfait pour le traitement d\'une dent","20":"Verres et monture","21":"Verres et monture","22":"Autres postes optique","23":"Autres postes optique","24":"Autres postes optique","25":"Honoraires m\\u00e9dicaux","26":"Honoraires m\\u00e9dicaux","27":"Honoraires m\\u00e9dicaux","28":"Honoraires m\\u00e9dicaux","29":"Honoraires m\\u00e9dicaux","30":"Honoraires m\\u00e9dicaux","31":"Honoraires m\\u00e9dicaux","32":"Honoraires m\\u00e9dicaux","33":"Honoraires m\\u00e9dicaux","34":"Honoraires m\\u00e9dicaux","35":"Imagerie m\\u00e9dicale","36":"Imagerie m\\u00e9dicale","37":"Analyses et examens de laboratoire","38":"Honoraires param\\u00e9dicaux","39":"M\\u00e9dicaments","40":"Mat\\u00e9riel M\\u00e9dical","41":"Mat\\u00e9riel M\\u00e9dical","42":"\\u00c9quipements 100 % Sant\\u00e9 * ( Classe 1 )","43":"Adulte ( B\\u00e9n\\u00e9ficiaire de 21 ans ou plus ) par oreille et par p\\u00e9riode de 4 ans","44":"Enfant ( B\\u00e9n\\u00e9ficiaire de moins de 21 ans ) par oreille et par p\\u00e9riode de 4 ans","45":" Accessoires, entretien et piles pris en charge par la S\\u00e9curit\\u00e9 sociale","46":"Frais de Transport","47":"Frais de Transport","48":"Frais de Transport","49":"Frais de Transport","50":"Accompagnement psychologique jusqu\'\\u00e0 8 s\\u00e9ances rembours\\u00e9es chez un psychologue\\npartenaire","51":"Assur\\u00e9 en affection de Longue Dur\\u00e9e","52":"Assur\\u00e9 en affection de Longue Dur\\u00e9e","53":"M\\u00e9decine non conventionnelle","54":"Pr\\u00e9vention","55":"Pr\\u00e9vention","56":"Pr\\u00e9vention"},"Garantie 2":{"0":"dans un \\u00e9tablissement conventionn\\u00e9 : ( frais de salle d\'op\\u00e9ration - pharmacie\\n- autres frais m\\u00e9dicaux et chirurgicaux frais de lit d\'accompagnant\\n\\/ enfants de moins de 12 ans )","1":"dans un \\u00e9tablissement non conventionn\\u00e9 : ( frais de salle d\'op\\u00e9ration - pharmacie\\n- autres frais m\\u00e9dicaux et chirurgicaux - frais de lit d\'accompagnant\\n\\/ enfants de moins de 12 ans )","2":"M\\u00e9decins adh\\u00e9rents \\u00e0 l\'un des DPTAM","3":"M\\u00e9decins non adh\\u00e9rents \\u00e0 l\'un des DPTAM","4":"Forfait journalier hospitalier","5":"Forfait actes lourds","6":"Chambre particuli\\u00e8re ( y compris maternit\\u00e9 )","7":"Frais d\'h\\u00e9bergement dans les EHPAD et les USLD sous r\\u00e9serve de l\'existence\\nd\'un tarif journalier de soins pris en charge par le r\\u00e9gime obligatoire","8":"Consultations , soins courants , radiologie , chirurgie et parodontologie ,\\npris en charge par la S\\u00e9curit\\u00e9 sociale","9":"\\u00e0 tarifs libres ***","10":"Couronnes provisoires","11":"Interm\\u00e9diaires de bridge ( y compris sur implants )","12":"R\\u00e9paration couronne ou rebasage de proth\\u00e8se adjointe","13":"Couronne non rembours\\u00e9e sur dent vivante ou non d\\u00e9labr\\u00e9e","14":"Scanner pr\\u00e9 - implantaire","15":"Actes de parodontie non rembours\\u00e9s","16":"Goutti\\u00e8re anti - bruxisme ( HBLD018 )","17":"Pilier de bridge non rembours\\u00e9","18":"Orthop\\u00e9die dento - faciale prise en charge par la S\\u00e9curit\\u00e9 sociale","19":"Implantologie dentaire Forfait pour le traitement d\'une dent","20":"\\u00c9quipements 100 % Sant\\u00e9 *\\nPrestation d\'adaptation ( Classe A et B )","21":"Grille Optique\\nVerres et monture \\u00e0 tarifs libres","22":"Les lentilles de contact correctrices prises en charge ou non par la S\\u00e9curit\\u00e9 sociale ,\\npar ann\\u00e9e civile et par b\\u00e9n\\u00e9ficiaire","23":"Au - del\\u00e0 , seules les lentilles prises en charge par la S\\u00e9curit\\u00e9 sociale seront rembour\\ns\\u00e9es \\u00e0 hauteur de","24":"L\'op\\u00e9ration de la myopie ( hyperm\\u00e9tropie ) , astigmatisme , presbytie et pose\\ndes implants intraoculaires par b\\u00e9n\\u00e9ficiaire et par ann\\u00e9e civile","25":"Consultation d\'un m\\u00e9decin g\\u00e9n\\u00e9raliste adh\\u00e9rent \\u00e0 l\'un des DPTAM","26":"Consultation d\'un m\\u00e9decin g\\u00e9n\\u00e9raliste non adh\\u00e9rent \\u00e0 l\'un des DPTAM","27":"Consultation d\'un m\\u00e9decin sp\\u00e9cialiste adh\\u00e9rent \\u00e0 l\'un des DPTAM","28":"Consultation d\'un m\\u00e9decin sp\\u00e9cialiste non adh\\u00e9rent \\u00e0 l\'un des DPTAM","29":"Visite d\'un m\\u00e9decin g\\u00e9n\\u00e9raliste adh\\u00e9rent \\u00e0 l\'un des DPTAMV","30":"Visite d\'un m\\u00e9decin g\\u00e9n\\u00e9raliste non adh\\u00e9rent \\u00e0 l\'un des DPTAM","31":"Visite d\'un m\\u00e9decin sp\\u00e9cialiste adh\\u00e9rent \\u00e0 l\'un des DPTAM","32":"Visite d\'un m\\u00e9decin sp\\u00e9cialiste non adh\\u00e9rent \\u00e0 l\'un des DPTAM","33":"Actes techniques m\\u00e9dicaux et actes de chirurgie pratiqu\\u00e9s par un m\\u00e9decin adh\\u00e9\\nrent \\u00e0 l\'un des DPTAM","34":"Actes techniques m\\u00e9dicaux et actes de chirurgie pratiqu\\u00e9s par un m\\u00e9decin non\\nadh\\u00e9rent \\u00e0 l\'un des DPTAM","35":"Actes d\'imagerie , \\u00e9chographies et dopplers pratiqu\\u00e9s\\npar un m\\u00e9decin adh\\u00e9rent \\u00e0 l\'un des DPTAM","36":"Actes d\'imagerie , \\u00e9chographies et dopplers pratiqu\\u00e9s\\npar un m\\u00e9decin non adh\\u00e9rent \\u00e0 l\'un des DPTAM","37":"Pris en charge par la S\\u00e9curit\\u00e9 sociale","38":"Actes pratiqu\\u00e9s par les auxiliaires m\\u00e9dicaux : les infirmiers ,\\nles masseurs kin\\u00e9sith\\u00e9rapeutes , les orthophonistes , les orthoptistes\\net les p\\u00e9dicures - podologues ...","39":"M\\u00e9dicaments hom\\u00e9opathiques prescrits par un m\\u00e9decin","40":"Gros appareillage , proth\\u00e8ses capillaires et mammaires , pris en charge\\npar la S\\u00e9curit\\u00e9 sociale .","41":"Petit appareillage","42":"\\u00c9quipements 100 % Sant\\u00e9 * ( Classe 1 )","43":"Adulte ( B\\u00e9n\\u00e9ficiaire de 21 ans ou plus ) par oreille et par p\\u00e9riode de 4 ans","44":"Enfant ( B\\u00e9n\\u00e9ficiaire de moins de 21 ans ) par oreille et par p\\u00e9riode de 4 ans","45":" Accessoires, entretien et piles pris en charge par la S\\u00e9curit\\u00e9 sociale","46":"Frais de Transport","47":"\\"Cures Thermales : frais m\\u00e9dicaux , de s\\u00e9jour et de transport\\npris en charge par la S\\u00e9curit\\u00e9 sociale\\"","48":"Consultation d\'un m\\u00e9decin adh\\u00e9rent \\u00e0 l\'un des DPTAM","49":"Consultation d\'un m\\u00e9decin non adh\\u00e9rent \\u00e0 l\'un des DPTAM","50":"Suppl\\u00e9ments factur\\u00e9s par \\u00e9tablissement thermal ( bain de boue et douche ) , s\'ils figurent sur la facture\\nde l\'\\u00e9tablissement et sont pris en charge par la S\\u00e9curit\\u00e9 sociale","51":"Accompagnement psychologique jusqu\'\\u00e0 8 s\\u00e9ances rembours\\u00e9es chez un psychologue\\npartenaire","52":"M\\u00e9dicaments ou traitements de confort non rembours\\u00e9s par la S\\u00e9curit\\u00e9 sociale sur prescription m\\u00e9dicale\\ndans le cas d\'un Assur\\u00e9 en Affection Longue dur\\u00e9e ( reconnu comme tel par la S\\u00e9curit\\u00e9 sociale \' ) Forfait\\npar an et par b\\u00e9n\\u00e9ficiaire","53":"Forfait actes th\\u00e9rapeutiques pour les actes cit\\u00e9s ci - apr\\u00e8s ,","54":"Vaccins non pris en charge par la S\\u00e9curit\\u00e9 sociale ( hors vaccin antigrippal ) , prescrits par un m\\u00e9decin dans\\nles conditions pr\\u00e9vues par leur autorisation de mise sur le march\\u00e9 , par ann\\u00e9e civile et par b\\u00e9n\\u00e9ficiaire","55":"Vaccin antigrippal non pris en charge par la S\\u00e9curit\\u00e9 sociale","56":"M\\u00e9dicaments et produits \\u00e0 base de nicotine , non pris en charge par la S\\u00e9curit\\u00e9 sociale , prescrits par\\nun m\\u00e9decin dans un but de sevrage tabagique , avec autorisation de mise sur le march\\u00e9 ou norme\\nAfnor , par ann\\u00e9e civile et par b\\u00e9n\\u00e9ficiaire"},"Formule SOCLE + (Responsable)":{"0":"100 % FR - MR","1":"95 % FR - MR","2":"220 % BR - MR","3":"200 % BR - MR","4":"100 % Forfait","5":"100 % Forfait","6":"100 % FR dans la limite\\nde 300 Euros \\/ jour","7":"200 % BR - MR","8":"400 % BR - MR","9":"400 % BR - MR","10":"60 Euros - MR \\/ dent","11":"Forfait de 100 Euros par inter \\u00e0 comp\\nter du 2\\u00e8me et sur ceux non rembours\\u00e9s\\n\\u00e0 compter du 1er","12":"50 Euros","13":"300 Euros \\/ couronne","14":"100 Euros","15":"80 Euros \\/ s\\u00e9ance","16":"100 Euros","17":"300 Euros \\/ pilier","18":"400 % BR - MR","19":"600 Euros","20":"100 % PLV - MR","21":"Grille optique par \\u00e9quipement","22":"700 Euros","23":"100 % TM","24":"610 Euros \\/ \\u0153il","25":"220 % BR - MR","26":"200 % BR - MR","27":"220 % BR - MR","28":"200 % BR - MR","29":"220 % BR - MR","30":"200 % BR - MR","31":"220 % BR - MR","32":"200 % BR - MR","33":"220 % BR - MR","34":"200 % BR - MR","35":"220 % BR - MR","36":"200 % BR - MR","37":"200 % BR - MR","38":"200 % BR - MR","39":"60 Euros \\/ an \\/ b\\u00e9n\\u00e9ficiaire","40":"300 % BR - MR","41":"200 % BR - MR","42":"100 % PLV - MR","43":"1600 Euros - MR","44":"1700 Euros - MR","45":"200 % BR - MR","46":"200 % BR -MR","47":"200 % BR - MR","48":"220 % BR - MR","49":"200 % BR - MR","50":"100 % BR","51":"100 % BR - MR","52":"200 Euros","53":"500 Euros \\/ an \\/ Famille","54":"100 % FR dans la limite de 200 Euros","55":"100 % FR","56":"100 % FR dans la limite de 30 Euros"},"Formule SOCLE +\\n& OPTION 1 (Option non responsable)":{"0":"100 % FR - MR","1":"95 % FR - MR","2":"520 % BR - MR","3":"500 % BR - MR","4":"100 % Forfait","5":"100 % Forfait","6":"100 % FR dans la limite\\nde 300 Euros \\/ jour","7":"200 % BR - MR","8":"650 % BR - MR","9":"650 % BR - MR","10":"150 Euros - MR \\/ dent","11":"Forfait de 200 Euros par inter \\u00e0 compter\\ndu 2\\u00e8me et sur ceux non rembours\\u00e9s \\u00e0\\ncompter du 1er","12":"120 Euros","13":"400 Euros \\/ couronne","14":"150 Euros","15":"250 Euros \\/ s\\u00e9ance","16":"250 Euros","17":"400 Euros \\/ pilier","18":"400 % BR - MR","19":"950 Euros","20":"100 % PLV - MR","21":"Grille optique par \\u00e9quipement\\n+ forfait suppl\\u00e9mentaire\\nde 400 Euros \\/ an \\/ b\\u00e9n\\u00e9ficiaire","22":"700 Euros","23":"100 % TM","24":"610 Euros \\/ \\u0153il","25":"270 % BR - MR","26":"250 % BR - MR","27":"270 % BR - MR","28":"250 % BR - MR","29":"270 % BR - MR","30":"250 % BR - MR","31":"270 % BR - MR","32":"250 % BR - MR","33":"270 % BR - MR","34":"250 % BR - MR","35":"270 % BR - MR","36":"250 % BR - MR","37":"250 % BR - MR","38":"250 % BR - MR","39":"60 Euros \\/ an \\/ b\\u00e9n\\u00e9ficiaire","40":"300 % BR - MR","41":"250 % BR - MR","42":"100 % PLV - MR","43":"1600 Euros - MR","44":"1700 Euros - MR","45":"200 % BR - MR","46":"250 % BR -MR","47":"250 % BR - MR","48":"220 % BR - MR","49":"200 % BR - MR","50":"100 % BR","51":"100 % BR - MR","52":"200 Euros","53":"500 Euros \\/ an \\/ Famille","54":"100 % FR dans la limite de 200 Euros","55":"100 % FR","56":"100 % FR dans la limite de 30 Euros"},"Formule SOCLE +\\n& OPTION 2 (Option non responsable)":{"0":"100 % FR - MR","1":"95 % FR - MR","2":"520 % BR - MR","3":"500 % BR - MR","4":"100 % Forfait","5":"100 % Forfait","6":"100 % FR dans la limite\\nde 300 Euros \\/ jour","7":"200 % BR - MR","8":"650 % BR - MR","9":"650 % BR - MR","10":"150 Euros - MR \\/ dent","11":"Forfait de 200 Euros par inter \\u00e0 compter\\ndu 2\\u00e8me et sur ceux non rembours\\u00e9s \\u00e0\\ncompter du 1er","12":"120 Euros","13":"400 Euros \\/ couronne","14":"150 Euros","15":"250 Euros \\/ s\\u00e9ance","16":"250 Euros","17":"400 Euros \\/ pilier","18":"500 % BR - MR","19":"950 Euros","20":"100 % PLV - MR","21":"Grille optique par \\u00e9quipement\\n+ forfait suppl\\u00e9mentaire\\nde 450 Euros \\/ an \\/ b\\u00e9n\\u00e9ficiaire","22":"700 Euros","23":"100 % TM","24":"750 Euros \\/ cil","25":"320 % BR - MR","26":"300 % BR - MR","27":"320 % BR - MR","28":"300 % BR - MR","29":"320 % BR - MR","30":"300 % BR - MR","31":"320 % BR - MR","32":"300 % BR - MR","33":"320 % BR - MR","34":"300 % BR - MR","35":"320 % BR - MR","36":"300 % BR - MR","37":"300 % BR - MR","38":"300 % BR - MR","39":"60 Euros \\/ an \\/ b\\u00e9n\\u00e9ficiaire","40":"400 % BR - MR","41":"300 % BR - MR","42":"100 % PLV - MR","43":"1600 Euros - MR","44":"1700 Euros - MR","45":"200 % BR - MR","46":"300 % BR - MR","47":"300 % BR - MR","48":"220 % BR - MR","49":"200 % BR - MR","50":"100 % BR","51":"100 % BR - MR","52":"250 Euros","53":"500 Euros \\/ an \\/ Famille","54":"100 % FR dans la limite de 200 Euros","55":"100 % FR","56":"100 % FR dans la limite de 30 Euros"},"Formule SOCLE + \\n& OPTION 3 (Option non responsable)":{"0":"100 % FR - MR","1":"95 % FR - MR","2":"620 % BR - MR","3":"600 % BR - MR","4":"100 % Forfait","5":"100 % Forfait","6":"100 % FR dans la limite\\nde 300 Euros \\/ \\/ jour","7":"200 % BR - MR","8":"800 % BR - MR","9":"800 % BR - MR","10":"150 Euros - MR \\/ dent","11":"Forfait de 300 Euros par inter \\u00e0 compter du\\n2\\u00e8me et sur ceux non rembours\\u00e9s \\u00e0 compter\\ndu 1er","12":"150 Euros","13":"500 Euros \\/ couronne","14":"200 Euros","15":"300 Euros \\/ s\\u00e9ance","16":"400 Euros","17":"500 Euros \\/ pilier","18":"600 % BR - MR","19":"1200 Euros","20":"100 % PLV - MR","21":"Grille optique par \\u00e9quipement\\n+ forfait suppl\\u00e9mentaire\\nde 550 Euros \\/ an \\/ b\\u00e9n\\u00e9ficiaire","22":"850 Euros","23":"100 % TM","24":"800 Euros \\/ \\u0153il","25":"620 % BR - MR","26":"600 % BR - MR","27":"620 % BR - MR","28":"600 % BR - MR","29":"620 % BR - MR","30":"600 % BR - MR","31":"620 % BR - MR","32":"600 % BR - MR","33":"620 % BR - MR","34":"600 % BR - MR","35":"520 % BR - MR","36":"500 % BR - MR","37":"500 % BR - MR","38":"500 % BR - MR","39":"60 Euros \\/ an \\/ b\\u00e9n\\u00e9ficiaire","40":"400 % BR - MR","41":"500 % BR - MR","42":"Option non responsable","43":"2300 Euros - MR","44":"2500 Euros - MR","45":"500 % BR - MR","46":"500 % BR -MR","47":"500 % BR -MR","48":"220 % BR - MR","49":"200 % BR - MR","50":"100 % BR","51":"100 % BR - MR","52":"300 Euros","53":"600 Euros \\/ an \\/ Famille","54":"100 % FR dans la limite de 200 Euros","55":"100 % FR","56":"100 % FR dans la limite de 30 Euros"}}'
tables={'AGIPI':'.\\comparaison_embedding_openai_small_model\\AGIPI_with_id_LSN.xlsx',
        'Henner':'.\\comparaison_embedding_openai_small_model\\Henner_with_id_LSN.xlsx',
        'amafi_socle':'.\\comparaison_embedding_openai_small_model\\amafi_socle_with_id_LSN.xlsx',
        'BNP':'.\\comparaison_embedding_openai_small_model\\BNP_with_id_LSN.xlsx',
        'generali':'.\\comparaison_embedding_openai_small_model\\generali_with_id_LSN.xlsx',
        }


tables_large={'AGIPI':'.\\comparaison_embedding_openai_large_model\\AGIPI_with_id_LSN.xlsx',
        'Henner':'.\\comparaison_embedding_openai_large_model\\Henner_with_id_LSN.xlsx',
        'amafi_socle':'.\\comparaison_embedding_openai_large_model\\amafi_socle_with_id_LSN.xlsx',
        'BNP':'.\\comparaison_embedding_openai_large_model\\BNP_with_id_LSN.xlsx',
        'generali':'.\\comparaison_embedding_openai_large_model\\generali_with_id_LSN.xlsx',
        }
def parcourir_df(df, colonnes, index):
    lignes = []
    for colonne in colonnes:
        lignes_colonne = df[df[colonne] == index]
        if not lignes_colonne.empty:
            lignes.extend(lignes_colonne.to_dict('records'))
    if len(lignes) == 0:
        return "Aucune ligne trouvée avec l'index spécifié dans les colonnes fournies."
    else:
        result_df = pd.DataFrame(lignes)
        # Afficher seulement les trois premières lignes s'il y en a
        if len(result_df) > 3:
            result_df = result_df.head(3)
        return result_df


def trouver_emplacement(phrase, df, colonne):
    try:
        index = df[df[colonne] == phrase].index[0]
        return index
    except IndexError:
        return "Phrase non trouvée dans la colonne spécifiée."
def main():
    # Titre de l'application
    st.title("Trouver la Garentie la plus proche ")
    df = pd.read_json(jp)
    
    choix= st.selectbox("Sélectionnez une option  :", ["","search using comparaison with small model embeddings","search using comparaison with small model embeddings"])
        


    if choix=="search using comparaison with small model embeddings":
            
        # Liste de noms
        liste_noms =[""]+ list(df['Type Garanties'].unique())
        
        
        # Afficher la liste de noms sous forme de formulaire
        nom_selectionne = st.selectbox("Sélectionnez un Type de Garentie :", liste_noms)
        liste_noms_1= [""]+list(df[df['Type Garanties'] == nom_selectionne]['Garantie 1'].unique())
        nom_selectionne_1= st.selectbox("Sélectionnez Garentie_1  :", liste_noms_1)
        liste_noms_2= [""]+list(df[df['Garantie 1'] == nom_selectionne_1]['Garantie 2'].unique())
        nom_selectionne_2= st.selectbox("Sélectionnez Garentie_2 :", liste_noms_2)
        ligne_selectionnee = df.loc[(df['Garantie 1'] == nom_selectionne_1) & (df['Garantie 2'] == nom_selectionne_2)]
        st.write(ligne_selectionnee)
        # Afficher le nom sélectionné
        if nom_selectionne!="" and nom_selectionne_1!="" and nom_selectionne_2!="" :
              index= ligne_selectionnee.index[0]
              #for table_name in tables.key():
              for nam_table in tables.keys():
                  dt=pd.read_excel(tables[nam_table])
                  dh=parcourir_df(dt,['id_lsn_1','id_lsn_2','id_lsn_3','id_lsn_4','id_lsn_5'],index)
                  st.write('for the inssurance company '+nam_table+' we have as similar guarantie :')
                  if type(dh)!=str:
                    if  df.shape[0]>=3:  
                        # Sélectionner les colonnes à partir de la troisième colonne
                        colonnes_selectionnees = list(dh.columns)[2:]

                        # Afficher les trois premières lignes
                        trois_premieres_lignes = dh[colonnes_selectionnees].head(3)                    
                        st.write(trois_premieres_lignes)
                    else:
                        st.write(dh[list(dh.columns)[2:]])
                  else:
                      st.write(dh)
    else:
            
        # Liste de noms
        liste_noms =[""]+ list(df['Type Garanties'].unique())
        
        
        # Afficher la liste de noms sous forme de formulaire
        nom_selectionne = st.selectbox("Sélectionnez un Type de Garentie :", liste_noms)
        liste_noms_1= [""]+list(df[df['Type Garanties'] == nom_selectionne]['Garantie 1'].unique())
        nom_selectionne_1= st.selectbox("Sélectionnez Garentie_1  :", liste_noms_1)
        liste_noms_2= [""]+list(df[df['Garantie 1'] == nom_selectionne_1]['Garantie 2'].unique())
        nom_selectionne_2= st.selectbox("Sélectionnez Garentie_2 :", liste_noms_2)
        ligne_selectionnee = df.loc[(df['Garantie 1'] == nom_selectionne_1) & (df['Garantie 2'] == nom_selectionne_2)]
        st.write(ligne_selectionnee)
        # Afficher le nom sélectionné
        if nom_selectionne!="" and nom_selectionne_1!="" and nom_selectionne_2!="" :
              index= ligne_selectionnee.index[0]
              #for table_name in tables.key():
              for nam_table in tables_large.keys():
                  dt=pd.read_excel(tables_large[nam_table])
                  dh=parcourir_df(dt,['id_lsn_1','id_lsn_2','id_lsn_3','id_lsn_4','id_lsn_5'],index)
                  st.write('for the inssurance company '+nam_table+' we have as similar guarantie :')
                  if type(dh)!=str:
                    if  df.shape[0]>=3:  
                        # Sélectionner les colonnes à partir de la troisième colonne
                        colonnes_selectionnees = list(dh.columns)[2:]

                        # Afficher les trois premières lignes
                        trois_premieres_lignes = dh[colonnes_selectionnees].head(3)                    
                        st.write(trois_premieres_lignes)
                    else:
                        st.write(dh[list(dh.columns)[2:]])
                  else:
                      st.write(dh)
                  
                  
                
              






if __name__ == "__main__":
    main()