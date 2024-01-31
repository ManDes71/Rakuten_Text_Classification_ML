
# ***PROJET RAKUTEN***  

## **1) Description du projet**  
**Description du problème**    

1.   Élément de liste
2.   Élément de liste


L'objectif de ce défi est la classification à grande échelle des données de produits multimodales (textes et images) en type de produits.  
Par exemple, dans le catalogue de Rakuten France, un produit avec une désignation "Grand Stylet Ergonomique Bleu Gamepad Nintendo Wii U - Speedlink Pilot Style" est associé à une image (image_938777978_product_201115110.jpg) et
à une description supplémentaire. Ce produit est catégorisé sous le code de produit 50.


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
pd.set_option('display.max_colwidth', 150)
```


```python
# lecture des données sources d'entrainement
df_feats=pd.read_csv('/content/Rakuten_Text_Classification_ML/X_train_update.csv')

# lecture des données cibles d'entrainement
df_target=pd.read_csv('/content/Rakuten_Text_Classification_ML/Y_train_CVw08PX.csv')

# création d'un dataframe globale -  jointure
df=df_feats.merge(df_target,on='Unnamed: 0',how='inner')
df.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>designation</th>
      <th>description</th>
      <th>productid</th>
      <th>imageid</th>
      <th>prdtypecode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Olivia: Personalisiertes Notizbuch / 150 Seiten / Punktraster / Ca Din A5 / Rosen-Design</td>
      <td>NaN</td>
      <td>3804725264</td>
      <td>1263597046</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Journal Des Arts (Le) N° 133 Du 28/09/2001 - L'art Et Son Marche Salon D'art Asiatique A Paris - Jacques Barrere - Francois Perrier - La Reforme D...</td>
      <td>NaN</td>
      <td>436067568</td>
      <td>1008141237</td>
      <td>2280</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Grand Stylet Ergonomique Bleu Gamepad Nintendo Wii U - Speedlink Pilot Style</td>
      <td>PILOT STYLE Touch Pen de marque Speedlink est 1 stylet ergonomique pour GamePad Nintendo Wii U.&lt;br&gt; Pour un confort optimal et une précision maxim...</td>
      <td>201115110</td>
      <td>938777978</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Peluche Donald - Europe - Disneyland 2000 (Marionnette À Doigt)</td>
      <td>NaN</td>
      <td>50418756</td>
      <td>457047496</td>
      <td>1280</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>La Guerre Des Tuques</td>
      <td>Luc a des id&amp;eacute;es de grandeur. Il veut organiser un jeu de guerre de boules de neige et s'arranger pour en &amp;ecirc;tre le vainqueur incontest&amp;...</td>
      <td>278535884</td>
      <td>1077757786</td>
      <td>2705</td>
    </tr>
  </tbody>
</table>

    
```python
import os
import cv2
import matplotlib.pyplot as plt

folder_path = '/content/Rakuten_Text_Classification_ML/images/image_test'


plt.figure(figsize=(10, 10))

for i in range(2, 5):
    filename = 'image_' + str(df.iloc[i, 4]) + "_product_" + str(df.iloc[i, 3]) + ".jpg"
    designation = df.iloc[i, 1]
    print("IMAGE ",i)
    print(designation)
    print(filename)
    img = cv2.imread(os.path.join(folder_path, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR en RGB
    plt.subplot(1, 3, i-1)
    plt.imshow(img)
    #
    plt.axis('off')

plt.show()

```

    IMAGE  2
    Grand Stylet Ergonomique Bleu Gamepad Nintendo Wii U - Speedlink Pilot Style
    image_938777978_product_201115110.jpg
    IMAGE  3
    Peluche Donald - Europe - Disneyland 2000 (Marionnette À Doigt)
    image_457047496_product_50418756.jpg
    IMAGE  4
    La Guerre Des Tuques
    image_1077757786_product_278535884.jpg
    


    
![png](images/ReadMe_ML_files/ReadMe_ML_10_1.png)
    


Ce notebook fait partie d'un ensemble de sous-projets dont le resultat représente le **projet Rakuten** que j'ai réalisé pour mon diplôme de data Scientist chez Datascientest.com.  
Ce projet consiste en la classification à grande échelle des données de         produits multimodales (textes et images) en type de produits.  
Ce repositery est la partie **Machine Learning** et ne traite que de la partie texte.   
Il utilise néanmoins la bibliothèque **Bibli_DataScience** commune à l'ensemble du projet.  
D'autres dépots viendront, à savoir  :


*   La partie image  traitée par des réseaux convolutifs
*   La partie texte  traitée par des réseaux récurrents
*   Une quatrième partie qui est une syntèse que j'ai présenté par l'outils Streamlit



Il existe d'autres produits avec des titres différents, des images
différentes et éventuellement des descriptions, qui appartiennent au même code
de produit.  
En utilisant ces informations sur les produits, ce
défi propose de modéliser un classificateur pour classer les produits dans leur code de produit correspondant.  

## **2) Introduction**   

**description des fichiers**

le but du projet est de prédire le code de type de chaque produit tel que défini dans le catalogue de Rakuten France.  
La catégorisation des annonces de produits se fait par le biais de la désignation, de la description (quand elle est présente) et des images.  
Les fichiers de données sont distribués ainsi :  
***X_train_update.csv*** : fichier d'entrée d'entraînement  
***Y_train_CVw08PX.csv*** : fichier de sortie d'entraînement  
***X_test_update.csv*** : fichier d'entrée de test  
Un fichier images.zip est également fourni, contenant toutes les images.  
La décompression de ce fichier fournira un dossier nommé "images" avec deux sous-dossiers nommés ***"image_train"*** et ***"image_test"***, contenant respectivement les images d'entraînement et de test.  
Pour ma part, ne participant pas au challenge Rakuten, je n'ai pas pas accès au fichier de sortie de test.  
Le fichier d’entrée de test est donc inutilisable.  
**X_train_update.csv** : fichier d'entrée d'entraînement :  
La première ligne du fichier d'entrée contient l'en-tête et les colonnes sont séparées par des virgules (",").  
Les colonnes sont les suivantes :  


*   **Un identifiant entier pour le produit**. Cet identifiant est utilisé pour associer le produit à son code de type de produit correspondant.
*   **Désignation** - Le titre du produit, un court texte résumant le produit
*   **Description** - Un texte plus détaillé décrivant le produit. Tous les marchands n'utilisent pas ce champ, il se peut donc que le champ de description contienne la valeur NaN pour de nombreux produits, afin de conserver l'originalité des données.
*   **productid** - Un identifiant unique pour le produit.
*   **imageid** - Un identifiant unique pour l'image associée au produit.
Les champs imageid et productid sont utilisés pour récupérer les images dans le dossier
d'images correspondant. Pour un produit donné, le nom du fichier image est :
image_imageid_product_productid.jpg ex : image_1263597046_product_3804725264.jpg  

**Y_train_CVw08PX.csv** : fichier de sortie d'entraînement :  
La première ligne des fichiers d'entrée contient l'en-tête et les colonnes sont séparées par des virgules (",").  
Les colonnes sont les suivantes :  
*  **Un identifiant entier pour le produit**. Cet identifiant est utilisé pour associer le produit à son
code de type de produit correspondant.
*  **prdtypecode** – Catégorie dans laquelle le produit est classé.

La liaison entre les fichiers se fait par une jointure sur l’identifiant entier présent sur les deux
fichiers.

## ***3) exploration du dataset.***  
Examinons la répartition  des codes produits :


```python
cat=df_target['prdtypecode'].unique()

plt.figure(figsize=(14, 8))
sns.countplot(data=df_target, x='prdtypecode', order = df_target['prdtypecode'].value_counts().index)
plt.xticks(rotation=90)  # Rotation des labels de l'axe x pour une meilleure lisibilité
plt.title("Distribution des prdtypecode")
plt.xlabel("Code produit (prdtypecode)")
plt.ylabel("Nombre d'occurrences")
plt.show()

print("il y a une grande disparité dans la répartition des classes !")

```


    
![png](images/ReadMe_ML_files/ReadMe_ML_15_0.png)
    


    il y a une grande disparité dans la répartition des classes !
    

# Proposition de nomenclature des classes ("prdtypecode")


```python
nomenclature=pd.read_csv('NOMENCLATURE.csv',header=0,encoding='utf-8',sep=';',index_col=0)
catdict=nomenclature.to_dict()['definition']
catdict
```




    {10: 'livres',
     40: 'jeux video pour pc et consoles',
     50: ' accesoires jeux video',
     60: 'consoles de jeux video',
     1140: 'produits derives “geeks” et figurines',
     1160: 'cartes collectionables',
     1180: 'figurines collectionables pour jeux de societe',
     1280: 'jouets, peluches, puppets',
     1281: 'jeux de societe/cartes',
     1300: 'Petites voitures (jouets) et maquettes',
     1301: 'accesoires pour petis enfants/bebes et mobilier de jeu (flechettes, billard, babyfoot)',
     1302: "jeux d'exterieur",
     1320: 'sacs pour femmes et accesore petite enfance',
     1560: 'Mobilier et produits decoration/rangement pour la maison',
     1920: 'linge de maison (cousins, rideaux, serviettes, nappes, draps)',
     1940: 'nouriture (cafes,infusions,conserves, epices,etc)',
     2060: 'lampes et accesoires decoration pour maison',
     2220: 'accesoires mascots/pets',
     2280: 'magazines',
     2403: 'livres et bds',
     2462: 'consoles de jeux video et jeux videos',
     2522: 'produits de papeterie et rangement bureau',
     2582: "mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)",
     2583: 'accesoires de piscine',
     2585: 'outillage et accesoires pour jardinage',
     2705: 'bds et livres',
     2905: 'Jeu En téléchargement'}



## Comparons les champs 'designation' et 'descriptions' :


```python
# Calcul de la moyenne des longueurs pour chaque colonne séparément
moyenne_designation = df_feats['designation'].str.len().mean()
moyenne_description = df_feats['description'].str.len().mean()

import matplotlib.pyplot as plt

categories = ['Designation', 'Description']

moyennes = [moyenne_designation, moyenne_description]

plt.figure(figsize=(16, 4))

plt.subplot(1, 2, 1)
plt.bar(categories, moyennes, color=['blue', 'green'])
plt.title('Moyenne des Longueurs des champs Designation et Description')
plt.xlabel('Catégories')
plt.ylabel('Moyenne des Longueurs')
plt.xticks(categories)

nb_designation = len(df_feats[~df_feats['designation'].isna()])
nb_description = len(df_feats['description'].unique())  # Assurez-vous que c'est bien 'description'

Nb = [nb_designation, nb_description]
plt.subplot(1, 2, 2)
plt.bar(categories, Nb, color=['red', 'yellow'])  # Choisir des couleurs différentes
plt.title('Valeurs non nulles des champs Designation et Description')
plt.xlabel('Catégories')
plt.ylabel('Nombre de produits')
plt.xticks(categories)

plt.show()

```


    
![png](images/ReadMe_ML_files/ReadMe_ML_20_0.png)
    


## Examinons les valeurs nulles et les doublons du champ 'designation':


```python
categories = ['Non nulles', 'Uniques']
nb_designation = len(df_feats[~df_feats['designation'].isna()])
nb_designation_u = len(df_feats['designation'].unique())

Nb = [nb_designation, nb_designation_u]

plt.figure(figsize=(8,4))  # Vous pouvez ajuster la taille selon vos besoins
plt.bar(categories,Nb, color=['blue', 'green'])  # Choisir des couleurs

plt.title('valeurs non nulles et unicité du champ  Designation')
plt.xlabel('Désignation')
plt.ylabel('Nombres de produits')
plt.xticks(categories)

plt.show()

```


    
![png](images/ReadMe_ML_files/ReadMe_ML_22_0.png)
    


## Examinons les valeurs nulles et les doublons du champ 'description'.


```python
categories = ['Non nulles', 'Uniques']
nb_description = len(df_feats[~df_feats['description'].isna()])
nb_description_u = len(df_feats['description'].unique())

Nb = [nb_description, nb_description_u]

plt.figure(figsize=(8, 4))
plt.bar(categories,Nb, color=['blue', 'green'])

plt.title('valeurs non nulles et unicité du champ  Description')
plt.xlabel('Description')
plt.ylabel('Nombres de produits')
plt.xticks(categories)

plt.show()
```


    
![png](images/ReadMe_ML_files/ReadMe_ML_24_0.png)
    


## ***4) Récupération du fichier df_langue.csv***


```python
df2=df.copy()
df_langue=pd.read_csv('/content/Rakuten_Text_Classification_ML/df_langue.csv')
df=df2.merge(df_langue.drop(['Unnamed: 0','prdtypecode'], axis=1),on='Id',how='inner')
df['status'] = df['descr_NaN'].apply(lambda x: 'SansDescrip' if x else 'AvecDescrip').astype(str)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 84916 entries, 0 to 84915
    Data columns (total 17 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Id              84916 non-null  int64  
     1   designation     84916 non-null  object 
     2   description     55116 non-null  object 
     3   productid       84916 non-null  int64  
     4   imageid         84916 non-null  int64  
     5   prdtypecode     84916 non-null  int64  
     6   PAYS_LANGUE     84916 non-null  object 
     7   RATIO_LANGUE    84916 non-null  float64
     8   ORIGINE_LANGUE  84916 non-null  object 
     9   pays_design     84916 non-null  object 
     10  Ratio_design    84916 non-null  float64
     11  pays_descr      55049 non-null  object 
     12  Ratio_descr     55049 non-null  float64
     13  design_long     84916 non-null  int64  
     14  descrip_long    55049 non-null  float64
     15  descr_NaN       84916 non-null  bool   
     16  status          84916 non-null  object 
    dtypes: bool(1), float64(4), int64(5), object(7)
    memory usage: 11.1+ MB
    

**Répartition des langues pour la colonne 'designation'**


```python
fig, axs = plt.subplots(1, 2, figsize=(15,5))
sns.countplot(x=df['pays_design'],ax=axs[0])
pays_principaux=df['pays_design'].value_counts()[:10]
print(pays_principaux.index)
sns.countplot(x=df[df['pays_design'].isin(pays_principaux.index)]['pays_design'],hue=df['status'],ax=axs[1])
plt.subplots_adjust( wspace=0.1,hspace=0.5)
plt.show()
```

    Index(['fr', 'en', 'de', 'nl', 'ca', 'it', 'ro', 'pt', 'es', 'no'], dtype='object')
    


    
![png](images/ReadMe_ML_files/ReadMe_ML_29_1.png)
    


**Répartition des langues pour la colonne 'description'**


```python
fig, axs = plt.subplots(1, 2, figsize=(15,5))
df_descrip=df.dropna(subset=['description'])
sns.countplot(x=df_descrip['pays_descr'],ax=axs[0])
pays_principaux=df['pays_descr'].value_counts()[:10]
print(pays_principaux.index)
plt.subplots_adjust( wspace=0.1,hspace=0.5)
sns.countplot(x=df[df['pays_descr'].isin(pays_principaux.index)]['pays_descr'],ax=axs[1])
plt.subplots_adjust( wspace=0.1,hspace=0.5)
plt.show()
```

    Index(['fr', 'en', 'de', 'ca', 'it', 'cy', 'pt', 'ro', 'es', 'vi'], dtype='object')
    


    
![png](images/ReadMe_ML_files/ReadMe_ML_31_1.png)
    


**Répartition des langues par catégorie (XX = le reste du monde)**


```python
import seaborn as sns
import matplotlib.pyplot as plt

# Modifier la taille de la figure ici
g = sns.FacetGrid(data=df, col='prdtypecode', col_wrap=3, sharex=False, sharey=False, height=4, aspect=1.2)
g.map_dataframe(sns.countplot, 'PAYS_LANGUE')
g.set_xticklabels(rotation=90)
g.set_xlabels('PAYS')
g.add_legend()

plt.show()

```


    
![png](images/ReadMe_ML_files/ReadMe_ML_33_0.png)
    


## ***5) Bags of words***


```python
stopwordFR = pd.read_csv("/content/Rakuten_Text_Classification_ML/stopwords_FR_02.csv")
Lcat=df_target.sort_values(by='prdtypecode')['prdtypecode'].unique()


nomenclature=pd.read_csv('/content/Rakuten_Text_Classification_ML/NOMENCLATURE.csv',header=0,encoding='utf-8',sep=';',index_col=0)
catdict=nomenclature.to_dict()['definition']

```


```python
df_top_40=pd.DataFrame()
vect=CountVectorizer(min_df=4, stop_words=stopwordFR['MOT'].tolist())
TailleCat ={}
for c in Lcat:
    df_cat=df[df['prdtypecode']==c]
    vect=vect.fit(df_cat['designation'])
    TailleCat[c]=len(vect.vocabulary_)
    print("Catégorie : ",c,"   Nombre de mots : ",  TailleCat[c])
    bag_of_words=vect.transform(df_cat['designation'])
    word_occurrences = np.sum(bag_of_words, axis=0)
    words = vect.get_feature_names_out()
    word_occurrences_dict = dict(zip(words, word_occurrences.tolist()[0]))
    top_40_words = sorted(word_occurrences_dict.items(), key=lambda x: x[1], reverse=True)[:40]
    dfout=pd.DataFrame(top_40_words,columns=['mot','occurence'])
    #print(dfout['occurence'].head(40))
    dfout['occurence'] = dfout['occurence'][:-1]
    #print(dfout['occurence'].head(40))
    dfout['occurence'] = dfout['occurence']
    dfout['prdtypecode']=c
    for word, occurrences in top_40_words:
        print(f"{word}: {occurrences}")
    df_top_40=pd.concat([df_top_40,dfout])
```

 

**Matrice de semblarité (Pourcentage de mots en commun (parmi les 40 premiers))**


```python
MAT=pd.DataFrame(np.zeros((len(Lcat),len(Lcat))) ,index=Lcat, columns=Lcat)

for c1 in Lcat:
    df1=df_top_40[df_top_40['prdtypecode']==c1]
    res1=[x for x in (df1.mot) ]
    for c2 in Lcat:
        df2=df_top_40[df_top_40['prdtypecode']==c2]
        res2=[x for x in (df2.mot) ]
        res=[x for x in res1 if x  in res2]
        #print(c1,c2,res)
        MAT.loc[c1,c2]=len(res)/40*100
```


```python
plt.figure(figsize=(10,10))
sns.heatmap(MAT)
```




    <Axes: >




    
![png](images/ReadMe_ML_files/ReadMe_ML_39_1.png)
    


**Nous pouvons déjà distinguer des catégories qui risquent de poser des problémes** :   

**les catégories 40,50 et 2462**
1.   40: 'jeux video pour pc et consoles'
2.   50: ' accesoires jeux video'
3.   2462: 'consoles de jeux video et jeux videos'
**les catégories 1280 et 1281**
1.   1280: 'jouets, peluches, puppets',
2.   1281: 'jeux de societe/cartes',
**les catégories 10, 2280, 2403 et 2705**
1.   10: 'livres'
2.   2280: 'magazines'
3.   2403: 'livres et bds'
4.   2705: 'bds et livres'

##***Nuages de mots***  
***Liste des 40 mots les plus fréquents par categorie***


```python

vect=CountVectorizer(min_df=4, stop_words=stopwordFR['MOT'].tolist())
TailleCat ={}
for c in Lcat:
    df_cat=df[df['prdtypecode']==c]
    vect=vect.fit(df_cat['designation'])
    TailleCat[c]=len(vect.vocabulary_)
    #print("Catégorie : ",c,"   Nombre de mots : ",  TailleCat[c],"  Nombre de produits ",len(df_cat))
    bag_of_words=vect.transform(df_cat['designation'])
    word_occurrences = np.sum(bag_of_words, axis=0)
    words = vect.get_feature_names_out()
    word_occurrences_dict = dict(zip(words, word_occurrences.tolist()[0]))
    top_40_words = sorted(word_occurrences_dict.items(), key=lambda x: x[1], reverse=True)[:40]
    dfout=pd.DataFrame(top_40_words,columns=['mot','occurence'])
    dfout['prdtypecode']=c
    """
    for word, occurrences in top_40_words:
        print(f"{word}: {occurrences}")
    """
    df_top_40=pd.concat([df_top_40,dfout])
```

    
```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline

# Définir le calque du nuage des mots
wc = WordCloud(background_color="black", max_words=100, stopwords=stopwordFR['MOT'].tolist(), max_font_size=50, random_state=42)
```


```python
df_top_40.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2160 entries, 0 to 39
    Data columns (total 3 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   mot          2160 non-null   object 
     1   occurence    2133 non-null   float64
     2   prdtypecode  2160 non-null   int64  
    dtypes: float64(1), int64(1), object(1)
    memory usage: 67.5+ KB
    


```python
fig, axs = plt.subplots(9, 3, figsize=(15,23))
for c,ax in zip(Lcat,axs.flat):
    #print(c)
    df_cat=df_top_40[df_top_40['prdtypecode']==c]
    # Définir la variable text
    text = ""
    for mot in df_cat['mot'] :
        text += mot + " "
    #print(c,"Catégorie ",catdict[c] )
    wc.generate(text)           # "Calcul" du wordcloud
    ax.imshow(wc) # Affichage
    ax.set_title( catdict[c][:30])
plt.subplots_adjust( wspace=0.1,hspace=0.5)
plt.show()

```


    
![png](images/ReadMe_ML_files/ReadMe_ML_45_0.png)
    


##***6) Machine Leaning***


```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.
    

Ce notebook teste plusieurs modèles de Machine Learning

explication de la bibliothèque **ML_Datascience**  :   

J'ai construit tout le code sur un modèle objet.  
Chaque modèle fait partie d'une classe et hérite d'une classe générale **DS_ML**

*   SVC -> classe **ML_SVC**
*   LogisticRegression -> classe **ML_LogisticRegression**
*   RandomForestClassifier -> classe **ML_RandomForest**
*   GradientBoostingClassifier -> classe **ML_GradientBoosting**
*   XGBClassifier -> classe **ML_XGBClassifier**
*   MultinomialNB -> classe **ML_MultinomialNB**
*   DecisionTreeClassifier -> classe **ML_DecisionTreeClassifier**


Pour tous les modèles on utilise le même préprocessing

*préprocessing utilisé :*
 1. concaténation des champs "désignation" et "description"
 2. avec une expression régulière :  `r = re.compile(r"[a-zA-Z0-9âéè°]{2,}")`
	 on ne garde	que certains caractères et les mots d'au moins 2 caractères
 3.  on découpe chaque phrase en une liste de mots : word_tokenize
 4.  on met tous les mots de la phrase en minuscule
 5.  on retire les stopswords par langue en utilisant module nltk.corpus


    def clean_sentence(self,sentence,langue):
            if langue == 'en':
                SW = stopwords.words('english')
            elif langue == 'fr':
                SW = stopwords.words('french')
            elif langue == 'de':
                SW = stopwords.words('german')
            elif langue == 'ca':
                SW = stopwords.words('french')
            elif langue == 'nl':
                SW = stopwords.words('dutch')
            elif langue == 'it':
                SW = stopwords.words('italian')
            elif langue == 'es':
                SW = stopwords.words('spanish')
            else:
                SW = stopwords.words('french')
             # Pour chaque mot de la phrase (dans l'ordre inverse)
            for i, word in reversed(list(enumerate(sentence))):
                    # Si le mot est un stopword
                    if word in SW :
                        # On l'enlève de l'artikle
                        sentence.pop(i)
            return sentence   

 6. on retire les stopswords issus d'une liste que j'ai personnalisée

> Au final on envoie au modèle une liste de mots caractéristiques issus
> des champs désignations et descriptions


```python
import Bibli_DataScience_3_1 as ds
import ML_DataScience as ml
import imp
imp.reload(ds)
imp.reload(ml)
```

### ***Modèle SVC***

la classe **ML_SVC** utilise un pipeline :

text_clf = Pipeline([
             ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),
                ('clf', SVC(C=10,class_weight=None,kernel='rbf',probability=True)), ])


```python
svc = ml.ML_SVC("Mon_Modele_SVC")
df_feat = svc.get_df_feats()
df = svc.get_DF()

```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    preprocessing ...


```python
df[['Id','designation','description','productid','imageid','PAYS_LANGUE','RATIO_LANGUE','phrases','prdtypecode']].head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>designation</th>
      <th>description</th>
      <th>productid</th>
      <th>imageid</th>
      <th>PAYS_LANGUE</th>
      <th>RATIO_LANGUE</th>
      <th>phrases</th>
      <th>prdtypecode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Olivia: Personalisiertes Notizbuch / 150 Seiten / Punktraster / Ca Din A5 / Rosen-Design</td>
      <td>NaN</td>
      <td>3804725264</td>
      <td>1263597046</td>
      <td>de</td>
      <td>0.99</td>
      <td>[olivia, personalisiertes, notizbuch, 150, seiten, punktraster, ca, din, a5, rosen, design]</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Journal Des Arts (Le) N° 133 Du 28/09/2001 - L'art Et Son Marche Salon D'art Asiatique A Paris - Jacques Barrere - Francois Perrier - La Reforme D...</td>
      <td>NaN</td>
      <td>436067568</td>
      <td>1008141237</td>
      <td>fr</td>
      <td>0.99</td>
      <td>[journal, arts, n°, 133, 28, 09, 2001, art, marche, salon, art, asiatique, paris, jacques, barrere, francois, perrier, reforme, ventes, encheres, ...</td>
      <td>2280</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Grand Stylet Ergonomique Bleu Gamepad Nintendo Wii U - Speedlink Pilot Style</td>
      <td>PILOT STYLE Touch Pen de marque Speedlink est 1 stylet ergonomique pour GamePad Nintendo Wii U.&lt;br&gt; Pour un confort optimal et une précision maxim...</td>
      <td>201115110</td>
      <td>938777978</td>
      <td>fr</td>
      <td>0.99</td>
      <td>[grand, stylet, ergonomique, bleu, gamepad, nintendo, wii, speedlink, pilot, style, pilot, style, touch, pen, marque, speedlink, stylet, ergonomiq...</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Peluche Donald - Europe - Disneyland 2000 (Marionnette À Doigt)</td>
      <td>NaN</td>
      <td>50418756</td>
      <td>457047496</td>
      <td>de</td>
      <td>0.71</td>
      <td>[peluche, donald, europe, disneyland, 2000, marionnette, doigt]</td>
      <td>1280</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>La Guerre Des Tuques</td>
      <td>Luc a des id&amp;eacute;es de grandeur. Il veut organiser un jeu de guerre de boules de neige et s'arranger pour en &amp;ecirc;tre le vainqueur incontest&amp;...</td>
      <td>278535884</td>
      <td>1077757786</td>
      <td>ca</td>
      <td>0.99</td>
      <td>[guerre, tuques, luc, id, grandeur, veut, organiser, jeu, guerre, boules, neige, arranger, vainqueur, incontest, sophie, chambarde, plans]</td>
      <td>2705</td>
    </tr>
  </tbody>
</table>


```python
svc.fit_modele(savefics=True,Train="Save")
```

L'heure au début de l'entraînement était :  2024-01-27 08:37:22.371312  
L'heure à la fin de l'entraînement était :  2024-01-27 11:59:04.704117  
F1 Score:  0.8294178505558132  
Accuracy:  0.8256594441827603  
La durée de l'entraînement était :  3:21:42.332805  


```python
y_orig = svc.get_y_orig()
y_pred = svc.get_y_pred()
_,_ = ds.get_classification_report(y_orig, y_pred)
```

    Précision de la prédiction: 82.56594441827603 %
    Evaluation détaillée de la Classification par RDF :
     
                   precision    recall  f1-score   support
    
              10       0.41      0.72      0.52       623
              40       0.76      0.68      0.72       502
              50       0.79      0.82      0.80       336
              60       0.99      0.81      0.89       166
            1140       0.81      0.79      0.80       534
            1160       0.94      0.93      0.93       791
            1180       0.93      0.58      0.71       153
            1280       0.70      0.73      0.71       974
            1281       0.67      0.55      0.61       414
            1300       0.94      0.94      0.94      1009
            1301       0.97      0.86      0.91       161
            1302       0.89      0.78      0.83       498
            1320       0.86      0.79      0.82       648
            1560       0.82      0.83      0.82      1015
            1920       0.92      0.90      0.91       861
            1940       0.99      0.81      0.89       161
            2060       0.81      0.81      0.81       999
            2220       0.90      0.78      0.83       165
            2280       0.74      0.85      0.79       952
            2403       0.78      0.75      0.77       955
            2462       0.83      0.75      0.79       284
            2522       0.94      0.91      0.93       998
            2582       0.84      0.73      0.78       518
            2583       0.98      0.98      0.98      2042
            2585       0.82      0.79      0.81       499
            2705       0.80      0.68      0.74       552
            2905       0.99      0.95      0.97       174
    
        accuracy                           0.83     16984
       macro avg       0.84      0.80      0.82     16984
    weighted avg       0.84      0.83      0.83     16984
    
    


```python
ds.show_confusion_matrix(y_orig, y_pred)
```


    
![png](images/ReadMe_ML_files/ReadMe_ML_59_0.png)
    


>tableau récapitulatif:  
**pour chaque classe réelle**, les **5 premières classes prédites** et leurs probabilités


```python
df_cross = svc.get_df_cross()
Lcat=svc.get_cat()
catdict = svc.get_catdict()
ds.Afficher_repartition(df_cross,Lcat,catdict)
```

    10    ------    livres
      : 10,  : 71.91 % , livres
      : 2280,  : 11.24 % , magazines
      : 2403,  : 6.26 % , livres et bds
      : 2705,  : 5.94 % , bds et livres
      : 40,  : 1.28 % , jeux video pour pc et consoles
    40    ------    jeux video pour pc et consoles
      : 40,  : 68.33 % , jeux video pour pc et consoles
      : 10,  : 8.57 % , livres
      : 50,  : 5.38 % ,  accesoires jeux video
      : 1280,  : 3.19 % , jouets, peluches, poupees
      : 2462,  : 2.59 % , consoles de jeux video et jeux videos
    50    ------     accesoires jeux video
      : 50,  : 81.85 % ,  accesoires jeux video
      : 2462,  : 4.17 % , consoles de jeux video et jeux videos
      : 40,  : 4.17 % , jeux video pour pc et consoles
      : 10,  : 2.98 % , livres
      : 1140,  : 1.49 % , produits derives “geeks” et figurines
    60    ------    consoles de jeux video
      : 60,  : 80.72 % , consoles de jeux video
      : 2462,  : 7.23 % , consoles de jeux video et jeux videos
      : 50,  : 6.63 % ,  accesoires jeux video
      : 40,  : 3.61 % , jeux video pour pc et consoles
      : 10,  : 1.2 % , livres
    1140    ------    produits derives “geeks” et figurines
      : 1140,  : 78.84 % , produits derives “geeks” et figurines
      : 1280,  : 5.06 % , jouets, peluches, poupees
      : 10,  : 3.93 % , livres
      : 2403,  : 2.25 % , livres et bds
      : 40,  : 2.06 % , jeux video pour pc et consoles
    1160    ------    cartes collectionables
      : 1160,  : 93.05 % , cartes collectionables
      : 10,  : 2.78 % , livres
      : 2280,  : 1.14 % , magazines
      : 40,  : 0.88 % , jeux video pour pc et consoles
      : 2403,  : 0.63 % , livres et bds
    1180    ------    figurines collectionables pour jeux de societe
      : 1180,  : 58.17 % , figurines collectionables pour jeux de societe
      : 10,  : 14.38 % , livres
      : 1280,  : 7.84 % , jouets, peluches, poupees
      : 1140,  : 5.23 % , produits derives “geeks” et figurines
      : 1281,  : 3.92 % , jeux de societe/cartes
    1280    ------    jouets, peluches, poupees
      : 1280,  : 72.9 % , jouets, peluches, poupees
      : 1281,  : 6.78 % , jeux de societe/cartes
      : 1300,  : 4.83 % , Petites voitures (jouets) et maquettes
      : 1140,  : 4.31 % , produits derives “geeks” et figurines
      : 10,  : 2.87 % , livres
    1281    ------    jeux de societe/cartes
      : 1281,  : 55.31 % , jeux de societe/cartes
      : 1280,  : 18.84 % , jouets, peluches, poupees
      : 10,  : 8.21 % , livres
      : 40,  : 3.38 % , jeux video pour pc et consoles
      : 2705,  : 2.9 % , bds et livres
    1300    ------    Petites voitures (jouets) et maquettes
      : 1300,  : 94.05 % , Petites voitures (jouets) et maquettes
      : 1280,  : 3.07 % , jouets, peluches, poupees
      : 10,  : 1.29 % , livres
      : 2280,  : 0.59 % , magazines
      : 1160,  : 0.2 % , cartes collectionables
    1301    ------    accesoires pour petis enfants/bebes et mobilier de jeu (flechettes, billard, babyfoot)
      : 1301,  : 86.34 % , accesoires pour petis enfants/bebes et mobilier de jeu (flechettes, billard, babyfoot)
      : 1280,  : 4.35 % , jouets, peluches, poupees
      : 10,  : 3.11 % , livres
      : 40,  : 1.24 % , jeux video pour pc et consoles
      : 1320,  : 1.24 % , sacs pour femmes et accesore petite enfance
    1302    ------    jeux d'exterieur
      : 1302,  : 78.11 % , jeux d'exterieur
      : 1280,  : 8.03 % , jouets, peluches, poupees
      : 10,  : 3.41 % , livres
      : 1281,  : 1.81 % , jeux de societe/cartes
      : 2583,  : 1.2 % , accesoires de piscine
    1320    ------    sacs pour femmes et accesore petite enfance
      : 1320,  : 78.7 % , sacs pour femmes et accesore petite enfance
      : 1280,  : 4.94 % , jouets, peluches, poupees
      : 10,  : 4.17 % , livres
      : 2060,  : 3.55 % , lampes et accesoires decoration pour maison
      : 1920,  : 1.7 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
    1560    ------    Mobilier et produits decoration/rangement pour la maison
      : 1560,  : 83.15 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 5.52 % , lampes et accesoires decoration pour maison
      : 2582,  : 3.15 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 1920,  : 2.46 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 2585,  : 1.87 % , outillage et accesoires pour jardinage
    1920    ------    linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 1920,  : 90.13 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 1560,  : 3.72 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 3.14 % , lampes et accesoires decoration pour maison
      : 1320,  : 0.93 % , sacs pour femmes et accesore petite enfance
      : 1280,  : 0.58 % , jouets, peluches, poupees
    1940    ------    nouriture (cafes,infusions,conserves, epices,etc)
      : 1940,  : 81.37 % , nouriture (cafes,infusions,conserves, epices,etc)
      : 10,  : 13.04 % , livres
      : 2403,  : 1.24 % , livres et bds
      : 2280,  : 1.24 % , magazines
      : 2705,  : 0.62 % , bds et livres
    2060    ------    lampes et accesoires decoration pour maison
      : 2060,  : 81.38 % , lampes et accesoires decoration pour maison
      : 1560,  : 6.91 % , Mobilier et produits decoration/rangement pour la maison
      : 1920,  : 2.1 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 10,  : 1.6 % , livres
      : 1280,  : 1.5 % , jouets, peluches, poupees
    2220    ------    accesoires mascots/pets
      : 2220,  : 77.58 % , accesoires mascots/pets
      : 10,  : 4.24 % , livres
      : 1320,  : 3.64 % , sacs pour femmes et accesore petite enfance
      : 1280,  : 3.03 % , jouets, peluches, poupees
      : 2060,  : 2.42 % , lampes et accesoires decoration pour maison
    2280    ------    magazines
      : 2280,  : 84.87 % , magazines
      : 10,  : 8.3 % , livres
      : 2403,  : 4.52 % , livres et bds
      : 2705,  : 0.74 % , bds et livres
      : 1160,  : 0.63 % , cartes collectionables
    2403    ------    livres et bds
      : 2403,  : 74.66 % , livres et bds
      : 2280,  : 10.99 % , magazines
      : 10,  : 10.79 % , livres
      : 2705,  : 1.78 % , bds et livres
      : 1160,  : 0.31 % , cartes collectionables
    2462    ------    consoles de jeux video et jeux videos
      : 2462,  : 75.0 % , consoles de jeux video et jeux videos
      : 40,  : 8.45 % , jeux video pour pc et consoles
      : 50,  : 6.34 % ,  accesoires jeux video
      : 10,  : 3.17 % , livres
      : 2403,  : 1.76 % , livres et bds
    2522    ------    produits de papeterie et rangement bureau
      : 2522,  : 91.48 % , produits de papeterie et rangement bureau
      : 10,  : 1.6 % , livres
      : 2403,  : 1.3 % , livres et bds
      : 2060,  : 1.0 % , lampes et accesoires decoration pour maison
      : 1560,  : 0.9 % , Mobilier et produits decoration/rangement pour la maison
    2582    ------    mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 2582,  : 73.17 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 1560,  : 6.95 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 5.98 % , lampes et accesoires decoration pour maison
      : 2585,  : 4.83 % , outillage et accesoires pour jardinage
      : 1920,  : 1.74 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
    2583    ------    accesoires de piscine
      : 2583,  : 97.7 % , accesoires de piscine
      : 2585,  : 0.39 % , outillage et accesoires pour jardinage
      : 10,  : 0.34 % , livres
      : 1302,  : 0.34 % , jeux d'exterieur
      : 1320,  : 0.29 % , sacs pour femmes et accesore petite enfance
    2585    ------    outillage et accesoires pour jardinage
      : 2585,  : 79.16 % , outillage et accesoires pour jardinage
      : 1560,  : 4.21 % , Mobilier et produits decoration/rangement pour la maison
      : 10,  : 3.61 % , livres
      : 2582,  : 3.41 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 2583,  : 3.01 % , accesoires de piscine
    2705    ------    bds et livres
      : 2705,  : 68.48 % , bds et livres
      : 10,  : 21.74 % , livres
      : 2280,  : 4.71 % , magazines
      : 2403,  : 3.44 % , livres et bds
      : 40,  : 0.54 % , jeux video pour pc et consoles
    2905    ------    Jeu En téléchargement
      : 2905,  : 94.83 % , Jeu En téléchargement
      : 1281,  : 1.72 % , jeux de societe/cartes
      : 1280,  : 1.15 % , jouets, peluches, poupees
      : 40,  : 1.15 % , jeux video pour pc et consoles
      : 2585,  : 0.57 % , outillage et accesoires pour jardinage
    

    /content/Rakuten_Text_Classification_ML/Bibli_DataScience_3_1.py:170: FutureWarning: iteritems is deprecated and will be removed in a future version. Use .items instead.
      for index, value in s.iteritems():
    

### ***LogisticRegression***

la classe **ML_LogisticRegression** utilise un pipeline :


```
text_lr = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),
            ('clf', LogisticRegression(C=1,fit_intercept=True,solver='liblinear',penalty='l2',max_iter= 200)),
            ])
```


```python
lr = ml.ML_LogisticRegression("LogisticRegression")
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    

    preprocessing ...
    


```python
lr.fit_modele(savefics=True,Train="Save")
```

    L'heure au début de l'entraînement était :  2024-01-28 16:45:17.804395
    L'heure à la fin de l'entraînement était :  2024-01-28 16:46:27.795792
    F1 Score:  0.8018564958168756
    Accuracy:  0.8022845030617052
    


```python
y_orig = lr.get_y_orig()
y_pred = lr.get_y_pred()
_,_ = ds.get_classification_report(y_orig, y_pred)
```

    Précision de la prédiction: 80.22845030617052 %
    Evaluation détaillée de la Classification par RDF :
     
                   precision    recall  f1-score   support
    
              10       0.45      0.67      0.53       623
              40       0.74      0.59      0.66       502
              50       0.74      0.75      0.74       336
              60       0.98      0.76      0.85       166
            1140       0.78      0.76      0.77       534
            1160       0.88      0.93      0.90       791
            1180       0.88      0.51      0.64       153
            1280       0.68      0.61      0.64       974
            1281       0.74      0.48      0.58       414
            1300       0.84      0.95      0.89      1009
            1301       0.98      0.84      0.90       161
            1302       0.81      0.75      0.78       498
            1320       0.83      0.73      0.78       648
            1560       0.79      0.82      0.80      1015
            1920       0.90      0.92      0.91       861
            1940       0.95      0.75      0.84       161
            2060       0.78      0.78      0.78       999
            2220       0.89      0.72      0.80       165
            2280       0.69      0.87      0.77       952
            2403       0.76      0.75      0.75       955
            2462       0.77      0.70      0.74       284
            2522       0.89      0.91      0.90       998
            2582       0.82      0.71      0.76       518
            2583       0.95      0.98      0.97      2042
            2585       0.83      0.76      0.79       499
            2705       0.77      0.69      0.73       552
            2905       0.98      0.93      0.95       174
    
        accuracy                           0.80     16984
       macro avg       0.82      0.76      0.78     16984
    weighted avg       0.81      0.80      0.80     16984
    
    


```python
ds.show_confusion_matrix(y_orig, y_pred)
```


    
![png](images/ReadMe_ML_files/ReadMe_ML_67_0.png)
    


>tableau récapitulatif:  
**pour chaque classe réelle**, les **5 premières classes prédites** et leurs probabilités


```python
df_cross = lr.get_df_cross()
ds.Afficher_repartition(df_cross,Lcat,catdict)
```

    10    ------    livres
      : 10,  : 66.93 % , livres
      : 2280,  : 14.45 % , magazines
      : 2403,  : 7.38 % , livres et bds
      : 2705,  : 6.74 % , bds et livres
      : 40,  : 0.8 % , jeux video pour pc et consoles
    40    ------    jeux video pour pc et consoles
      : 40,  : 59.36 % , jeux video pour pc et consoles
      : 10,  : 10.36 % , livres
      : 50,  : 7.57 % ,  accesoires jeux video
      : 2280,  : 4.18 % , magazines
      : 1280,  : 3.78 % , jouets, peluches, poupees
    50    ------     accesoires jeux video
      : 50,  : 75.3 % ,  accesoires jeux video
      : 2462,  : 6.25 % , consoles de jeux video et jeux videos
      : 40,  : 3.27 % , jeux video pour pc et consoles
      : 1140,  : 2.68 % , produits derives “geeks” et figurines
      : 1560,  : 1.79 % , Mobilier et produits decoration/rangement pour la maison
    60    ------    consoles de jeux video
      : 60,  : 75.9 % , consoles de jeux video
      : 2462,  : 9.04 % , consoles de jeux video et jeux videos
      : 50,  : 7.83 % ,  accesoires jeux video
      : 40,  : 4.82 % , jeux video pour pc et consoles
      : 10,  : 1.2 % , livres
    1140    ------    produits derives “geeks” et figurines
      : 1140,  : 75.66 % , produits derives “geeks” et figurines
      : 1280,  : 4.49 % , jouets, peluches, poupees
      : 2280,  : 3.56 % , magazines
      : 10,  : 2.62 % , livres
      : 1160,  : 2.43 % , cartes collectionables
    1160    ------    cartes collectionables
      : 1160,  : 92.92 % , cartes collectionables
      : 10,  : 2.78 % , livres
      : 2280,  : 1.39 % , magazines
      : 40,  : 0.76 % , jeux video pour pc et consoles
      : 2403,  : 0.63 % , livres et bds
    1180    ------    figurines collectionables pour jeux de societe
      : 1180,  : 50.98 % , figurines collectionables pour jeux de societe
      : 10,  : 14.38 % , livres
      : 1140,  : 7.84 % , produits derives “geeks” et figurines
      : 1280,  : 5.88 % , jouets, peluches, poupees
      : 1160,  : 3.92 % , cartes collectionables
    1280    ------    jouets, peluches, poupees
      : 1280,  : 61.09 % , jouets, peluches, poupees
      : 1300,  : 15.3 % , Petites voitures (jouets) et maquettes
      : 1140,  : 5.34 % , produits derives “geeks” et figurines
      : 1302,  : 2.77 % , jeux d'exterieur
      : 10,  : 2.57 % , livres
    1281    ------    jeux de societe/cartes
      : 1281,  : 47.83 % , jeux de societe/cartes
      : 1280,  : 21.26 % , jouets, peluches, poupees
      : 10,  : 6.76 % , livres
      : 1160,  : 4.83 % , cartes collectionables
      : 2705,  : 2.9 % , bds et livres
    1300    ------    Petites voitures (jouets) et maquettes
      : 1300,  : 94.95 % , Petites voitures (jouets) et maquettes
      : 1280,  : 1.78 % , jouets, peluches, poupees
      : 10,  : 0.99 % , livres
      : 2280,  : 0.69 % , magazines
      : 2583,  : 0.4 % , accesoires de piscine
    1301    ------    accesoires pour petis enfants/bebes et mobilier de jeu (flechettes, billard, babyfoot)
      : 1301,  : 83.85 % , accesoires pour petis enfants/bebes et mobilier de jeu (flechettes, billard, babyfoot)
      : 10,  : 3.11 % , livres
      : 1280,  : 3.11 % , jouets, peluches, poupees
      : 40,  : 1.86 % , jeux video pour pc et consoles
      : 1320,  : 1.86 % , sacs pour femmes et accesore petite enfance
    1302    ------    jeux d'exterieur
      : 1302,  : 75.1 % , jeux d'exterieur
      : 1280,  : 8.03 % , jouets, peluches, poupees
      : 10,  : 2.81 % , livres
      : 2583,  : 1.81 % , accesoires de piscine
      : 1281,  : 1.81 % , jeux de societe/cartes
    1320    ------    sacs pour femmes et accesore petite enfance
      : 1320,  : 73.3 % , sacs pour femmes et accesore petite enfance
      : 2060,  : 5.25 % , lampes et accesoires decoration pour maison
      : 1280,  : 4.63 % , jouets, peluches, poupees
      : 10,  : 3.24 % , livres
      : 1560,  : 3.24 % , Mobilier et produits decoration/rangement pour la maison
    1560    ------    Mobilier et produits decoration/rangement pour la maison
      : 1560,  : 81.87 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 6.21 % , lampes et accesoires decoration pour maison
      : 2582,  : 3.05 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 1920,  : 2.96 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 2585,  : 1.67 % , outillage et accesoires pour jardinage
    1920    ------    linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 1920,  : 91.52 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 2060,  : 3.02 % , lampes et accesoires decoration pour maison
      : 1560,  : 2.32 % , Mobilier et produits decoration/rangement pour la maison
      : 1320,  : 0.93 % , sacs pour femmes et accesore petite enfance
      : 2583,  : 0.46 % , accesoires de piscine
    1940    ------    nouriture (cafes,infusions,conserves, epices,etc)
      : 1940,  : 75.16 % , nouriture (cafes,infusions,conserves, epices,etc)
      : 10,  : 11.18 % , livres
      : 2403,  : 2.48 % , livres et bds
      : 2522,  : 1.86 % , produits de papeterie et rangement bureau
      : 2280,  : 1.86 % , magazines
    2060    ------    lampes et accesoires decoration pour maison
      : 2060,  : 78.48 % , lampes et accesoires decoration pour maison
      : 1560,  : 6.81 % , Mobilier et produits decoration/rangement pour la maison
      : 1920,  : 3.8 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 2522,  : 1.6 % , produits de papeterie et rangement bureau
      : 1320,  : 1.3 % , sacs pour femmes et accesore petite enfance
    2220    ------    accesoires mascots/pets
      : 2220,  : 72.12 % , accesoires mascots/pets
      : 1320,  : 4.85 % , sacs pour femmes et accesore petite enfance
      : 2522,  : 4.24 % , produits de papeterie et rangement bureau
      : 1280,  : 3.64 % , jouets, peluches, poupees
      : 2582,  : 2.42 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
    2280    ------    magazines
      : 2280,  : 86.66 % , magazines
      : 10,  : 5.46 % , livres
      : 2403,  : 4.52 % , livres et bds
      : 1160,  : 1.16 % , cartes collectionables
      : 2705,  : 0.95 % , bds et livres
    2403    ------    livres et bds
      : 2403,  : 74.97 % , livres et bds
      : 2280,  : 12.15 % , magazines
      : 10,  : 8.27 % , livres
      : 2705,  : 1.88 % , bds et livres
      : 1160,  : 0.63 % , cartes collectionables
    2462    ------    consoles de jeux video et jeux videos
      : 2462,  : 70.07 % , consoles de jeux video et jeux videos
      : 50,  : 9.86 % ,  accesoires jeux video
      : 40,  : 6.34 % , jeux video pour pc et consoles
      : 2403,  : 3.52 % , livres et bds
      : 10,  : 2.82 % , livres
    2522    ------    produits de papeterie et rangement bureau
      : 2522,  : 91.48 % , produits de papeterie et rangement bureau
      : 2403,  : 1.5 % , livres et bds
      : 2060,  : 1.1 % , lampes et accesoires decoration pour maison
      : 10,  : 1.0 % , livres
      : 1560,  : 1.0 % , Mobilier et produits decoration/rangement pour la maison
    2582    ------    mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 2582,  : 70.85 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 1560,  : 9.07 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 5.79 % , lampes et accesoires decoration pour maison
      : 2585,  : 4.05 % , outillage et accesoires pour jardinage
      : 2583,  : 2.7 % , accesoires de piscine
    2583    ------    accesoires de piscine
      : 2583,  : 97.85 % , accesoires de piscine
      : 2060,  : 0.44 % , lampes et accesoires decoration pour maison
      : 1302,  : 0.34 % , jeux d'exterieur
      : 1320,  : 0.24 % , sacs pour femmes et accesore petite enfance
      : 1560,  : 0.2 % , Mobilier et produits decoration/rangement pour la maison
    2585    ------    outillage et accesoires pour jardinage
      : 2585,  : 75.55 % , outillage et accesoires pour jardinage
      : 2583,  : 5.41 % , accesoires de piscine
      : 1560,  : 4.81 % , Mobilier et produits decoration/rangement pour la maison
      : 2582,  : 2.81 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 2522,  : 2.61 % , produits de papeterie et rangement bureau
    2705    ------    bds et livres
      : 2705,  : 68.84 % , bds et livres
      : 10,  : 19.02 % , livres
      : 2280,  : 7.25 % , magazines
      : 2403,  : 3.08 % , livres et bds
      : 1320,  : 0.36 % , sacs pour femmes et accesore petite enfance
    2905    ------    Jeu En téléchargement
      : 2905,  : 93.1 % , Jeu En téléchargement
      : 1281,  : 2.3 % , jeux de societe/cartes
      : 2705,  : 1.72 % , bds et livres
      : 1280,  : 1.72 % , jouets, peluches, poupees
      : 2585,  : 0.57 % , outillage et accesoires pour jardinage
    

    /content/Rakuten_Text_Classification_ML/Bibli_DataScience_3_1.py:170: FutureWarning: iteritems is deprecated and will be removed in a future version. Use .items instead.
      for index, value in s.iteritems():
    

### ***RandomForestClassifier***

la classe **ML_RandomForest** utilise un pipeline :

```
text_forest = Pipeline([  
            ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),  
            ('clf', RandomForestClassifier(n_jobs=-1,random_state=321)),  
            ])
```


```python
forest = ml.ML_RandomForest("RandomForestClassifier")
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    

    preprocessing ...
    


```python
forest.fit_modele(savefics=True,Train="Save")
```

    L'heure au début de l'entraînement était :  2024-01-28 19:24:38.880681
    L'heure à la fin de l'entraînement était :  2024-01-28 19:33:16.875676
    F1 Score:  0.7932958883861291
    Accuracy:  0.7926283560998587
    


```python
y_orig = forest.get_y_orig()
y_pred = forest.get_y_pred()
_,_ = ds.get_classification_report(y_orig, y_pred)
```

    Précision de la prédiction: 79.26283560998587 %
    Evaluation détaillée de la Classification par RDF :
     
                   precision    recall  f1-score   support
    
              10       0.40      0.70      0.51       623
              40       0.78      0.61      0.68       502
              50       0.83      0.75      0.79       336
              60       0.99      0.81      0.89       166
            1140       0.77      0.78      0.78       534
            1160       0.89      0.92      0.90       791
            1180       0.88      0.54      0.67       153
            1280       0.68      0.61      0.64       974
            1281       0.68      0.51      0.59       414
            1300       0.84      0.93      0.88      1009
            1301       0.99      0.84      0.91       161
            1302       0.90      0.70      0.79       498
            1320       0.85      0.72      0.78       648
            1560       0.76      0.80      0.78      1015
            1920       0.89      0.90      0.90       861
            1940       0.95      0.76      0.84       161
            2060       0.74      0.79      0.77       999
            2220       0.94      0.53      0.68       165
            2280       0.76      0.80      0.78       952
            2403       0.75      0.75      0.75       955
            2462       0.71      0.81      0.76       284
            2522       0.87      0.89      0.88       998
            2582       0.81      0.66      0.73       518
            2583       0.91      0.98      0.94      2042
            2585       0.86      0.64      0.73       499
            2705       0.77      0.66      0.71       552
            2905       1.00      1.00      1.00       174
    
        accuracy                           0.79     16984
       macro avg       0.82      0.76      0.78     16984
    weighted avg       0.80      0.79      0.79     16984
    
    


```python
ds.show_confusion_matrix(y_orig, y_pred)
```


    
![png](images/ReadMe_ML_files/ReadMe_ML_75_0.png)
    



```python
df_cross = forest.get_df_cross()
ds.Afficher_repartition(df_cross,Lcat,catdict)
```

    10    ------    livres
      : 10,  : 70.14 % , livres
      : 2280,  : 9.47 % , magazines
      : 2403,  : 7.87 % , livres et bds
      : 2705,  : 5.14 % , bds et livres
      : 40,  : 2.89 % , jeux video pour pc et consoles
    40    ------    jeux video pour pc et consoles
      : 40,  : 60.96 % , jeux video pour pc et consoles
      : 10,  : 10.56 % , livres
      : 2462,  : 5.98 % , consoles de jeux video et jeux videos
      : 50,  : 4.38 % ,  accesoires jeux video
      : 2705,  : 2.79 % , bds et livres
    50    ------     accesoires jeux video
      : 50,  : 75.3 % ,  accesoires jeux video
      : 2462,  : 8.33 % , consoles de jeux video et jeux videos
      : 40,  : 2.38 % , jeux video pour pc et consoles
      : 1300,  : 2.08 % , Petites voitures (jouets) et maquettes
      : 10,  : 1.79 % , livres
    60    ------    consoles de jeux video
      : 60,  : 80.72 % , consoles de jeux video
      : 2462,  : 11.45 % , consoles de jeux video et jeux videos
      : 50,  : 4.82 % ,  accesoires jeux video
      : 40,  : 1.81 % , jeux video pour pc et consoles
      : 10,  : 0.6 % , livres
    1140    ------    produits derives “geeks” et figurines
      : 1140,  : 78.09 % , produits derives “geeks” et figurines
      : 10,  : 3.75 % , livres
      : 1280,  : 3.18 % , jouets, peluches, poupees
      : 2280,  : 2.62 % , magazines
      : 2403,  : 2.62 % , livres et bds
    1160    ------    cartes collectionables
      : 1160,  : 92.41 % , cartes collectionables
      : 10,  : 3.16 % , livres
      : 2280,  : 1.52 % , magazines
      : 40,  : 1.14 % , jeux video pour pc et consoles
      : 2403,  : 0.63 % , livres et bds
    1180    ------    figurines collectionables pour jeux de societe
      : 1180,  : 54.25 % , figurines collectionables pour jeux de societe
      : 10,  : 15.69 % , livres
      : 1140,  : 7.19 % , produits derives “geeks” et figurines
      : 1281,  : 5.23 % , jeux de societe/cartes
      : 1280,  : 4.58 % , jouets, peluches, poupees
    1280    ------    jouets, peluches, poupees
      : 1280,  : 61.19 % , jouets, peluches, poupees
      : 1300,  : 13.24 % , Petites voitures (jouets) et maquettes
      : 1140,  : 5.65 % , produits derives “geeks” et figurines
      : 1281,  : 4.41 % , jeux de societe/cartes
      : 10,  : 3.7 % , livres
    1281    ------    jeux de societe/cartes
      : 1281,  : 51.45 % , jeux de societe/cartes
      : 1280,  : 19.57 % , jouets, peluches, poupees
      : 10,  : 6.04 % , livres
      : 1160,  : 5.07 % , cartes collectionables
      : 2705,  : 2.66 % , bds et livres
    1300    ------    Petites voitures (jouets) et maquettes
      : 1300,  : 92.96 % , Petites voitures (jouets) et maquettes
      : 1280,  : 3.07 % , jouets, peluches, poupees
      : 10,  : 1.78 % , livres
      : 2583,  : 0.4 % , accesoires de piscine
      : 2280,  : 0.3 % , magazines
    1301    ------    accesoires pour petis enfants/bebes et mobilier de jeu (flechettes, billard, babyfoot)
      : 1301,  : 83.85 % , accesoires pour petis enfants/bebes et mobilier de jeu (flechettes, billard, babyfoot)
      : 1280,  : 3.11 % , jouets, peluches, poupees
      : 2583,  : 2.48 % , accesoires de piscine
      : 2522,  : 2.48 % , produits de papeterie et rangement bureau
      : 1320,  : 2.48 % , sacs pour femmes et accesore petite enfance
    1302    ------    jeux d'exterieur
      : 1302,  : 70.28 % , jeux d'exterieur
      : 1280,  : 7.63 % , jouets, peluches, poupees
      : 2583,  : 3.82 % , accesoires de piscine
      : 10,  : 2.41 % , livres
      : 1281,  : 2.41 % , jeux de societe/cartes
    1320    ------    sacs pour femmes et accesore petite enfance
      : 1320,  : 72.38 % , sacs pour femmes et accesore petite enfance
      : 1280,  : 5.09 % , jouets, peluches, poupees
      : 2060,  : 4.78 % , lampes et accesoires decoration pour maison
      : 10,  : 4.17 % , livres
      : 1560,  : 2.62 % , Mobilier et produits decoration/rangement pour la maison
    1560    ------    Mobilier et produits decoration/rangement pour la maison
      : 1560,  : 80.49 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 7.98 % , lampes et accesoires decoration pour maison
      : 2582,  : 3.05 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 1920,  : 2.86 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 2522,  : 0.99 % , produits de papeterie et rangement bureau
    1920    ------    linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 1920,  : 90.24 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 1560,  : 3.48 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 2.79 % , lampes et accesoires decoration pour maison
      : 2583,  : 0.58 % , accesoires de piscine
      : 2522,  : 0.58 % , produits de papeterie et rangement bureau
    1940    ------    nouriture (cafes,infusions,conserves, epices,etc)
      : 1940,  : 75.78 % , nouriture (cafes,infusions,conserves, epices,etc)
      : 10,  : 6.83 % , livres
      : 2583,  : 3.73 % , accesoires de piscine
      : 2522,  : 2.48 % , produits de papeterie et rangement bureau
      : 2403,  : 2.48 % , livres et bds
    2060    ------    lampes et accesoires decoration pour maison
      : 2060,  : 79.48 % , lampes et accesoires decoration pour maison
      : 1560,  : 7.51 % , Mobilier et produits decoration/rangement pour la maison
      : 1920,  : 3.0 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 2582,  : 1.7 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 2583,  : 1.3 % , accesoires de piscine
    2220    ------    accesoires mascots/pets
      : 2220,  : 53.33 % , accesoires mascots/pets
      : 1280,  : 8.48 % , jouets, peluches, poupees
      : 2060,  : 7.88 % , lampes et accesoires decoration pour maison
      : 2522,  : 6.67 % , produits de papeterie et rangement bureau
      : 2583,  : 4.85 % , accesoires de piscine
    2280    ------    magazines
      : 2280,  : 80.46 % , magazines
      : 10,  : 9.87 % , livres
      : 2403,  : 5.25 % , livres et bds
      : 2705,  : 1.16 % , bds et livres
      : 1160,  : 1.05 % , cartes collectionables
    2403    ------    livres et bds
      : 2403,  : 74.97 % , livres et bds
      : 10,  : 11.83 % , livres
      : 2280,  : 8.17 % , magazines
      : 2705,  : 1.57 % , bds et livres
      : 1160,  : 0.94 % , cartes collectionables
    2462    ------    consoles de jeux video et jeux videos
      : 2462,  : 81.34 % , consoles de jeux video et jeux videos
      : 40,  : 4.93 % , jeux video pour pc et consoles
      : 50,  : 2.82 % ,  accesoires jeux video
      : 10,  : 2.46 % , livres
      : 2403,  : 2.11 % , livres et bds
    2522    ------    produits de papeterie et rangement bureau
      : 2522,  : 88.88 % , produits de papeterie et rangement bureau
      : 2403,  : 1.7 % , livres et bds
      : 10,  : 1.6 % , livres
      : 1560,  : 1.4 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 1.3 % , lampes et accesoires decoration pour maison
    2582    ------    mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 2582,  : 66.02 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 1560,  : 11.58 % , Mobilier et produits decoration/rangement pour la maison
      : 2583,  : 5.6 % , accesoires de piscine
      : 2060,  : 5.41 % , lampes et accesoires decoration pour maison
      : 2585,  : 3.09 % , outillage et accesoires pour jardinage
    2583    ------    accesoires de piscine
      : 2583,  : 98.09 % , accesoires de piscine
      : 10,  : 0.29 % , livres
      : 2060,  : 0.29 % , lampes et accesoires decoration pour maison
      : 1280,  : 0.2 % , jouets, peluches, poupees
      : 1320,  : 0.15 % , sacs pour femmes et accesore petite enfance
    2585    ------    outillage et accesoires pour jardinage
      : 2585,  : 63.93 % , outillage et accesoires pour jardinage
      : 2583,  : 13.03 % , accesoires de piscine
      : 1560,  : 6.01 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 5.81 % , lampes et accesoires decoration pour maison
      : 2522,  : 3.01 % , produits de papeterie et rangement bureau
    2705    ------    bds et livres
      : 2705,  : 65.94 % , bds et livres
      : 10,  : 23.37 % , livres
      : 2403,  : 3.99 % , livres et bds
      : 2280,  : 3.8 % , magazines
      : 1320,  : 0.54 % , sacs pour femmes et accesore petite enfance
    2905    ------    Jeu En téléchargement
      : 2905,  : 100.0 % , Jeu En téléchargement
      : 1920,  : 0.0 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 2705,  : 0.0 % , bds et livres
      : 2585,  : 0.0 % , outillage et accesoires pour jardinage
      : 2583,  : 0.0 % , accesoires de piscine
    

    /content/Rakuten_Text_Classification_ML/Bibli_DataScience_3_1.py:170: FutureWarning: iteritems is deprecated and will be removed in a future version. Use .items instead.
      for index, value in s.iteritems():
    

### ***GradientBoosting***
la classe la classe **ML_GradientBoosting** utilise un pipeline :


```
text_gboost = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),  
            ('clf', GradientBoostingClassifier(criterion= 'squared_error',
              learning_rate= 0.1, loss= 'log_loss', max_depth = 18,max_features = 'sqrt')),  
            ])
```



```python
gboost  = ml.ML_GradientBoosting("GradientBoosting")
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    

    preprocessing ...
    


```python
gboost.fit_modele(savefics=True,Train="Save")
```

    L'heure au début de l'entraînement était :  2024-01-28 19:43:00.769945
    L'heure à la fin de l'entraînement était :  2024-01-28 19:53:13.431463
    F1 Score:  0.7584126603103837
    Accuracy:  0.7573009891662741
    


```python
y_orig = gboost.get_y_orig()
y_pred = gboost.get_y_pred()
_,_ = ds.get_classification_report(y_orig, y_pred)
```

    Précision de la prédiction: 75.73009891662741 %
    Evaluation détaillée de la Classification par RDF :
     
                   precision    recall  f1-score   support
    
              10       0.35      0.61      0.44       623
              40       0.68      0.52      0.59       502
              50       0.66      0.62      0.64       336
              60       0.85      0.72      0.78       166
            1140       0.75      0.72      0.73       534
            1160       0.87      0.89      0.88       791
            1180       0.53      0.49      0.51       153
            1280       0.63      0.63      0.63       974
            1281       0.55      0.39      0.46       414
            1300       0.88      0.91      0.90      1009
            1301       0.77      0.74      0.76       161
            1302       0.80      0.69      0.74       498
            1320       0.82      0.68      0.74       648
            1560       0.77      0.80      0.78      1015
            1920       0.90      0.89      0.90       861
            1940       0.66      0.57      0.61       161
            2060       0.76      0.77      0.76       999
            2220       0.63      0.42      0.51       165
            2280       0.68      0.77      0.72       952
            2403       0.65      0.72      0.69       955
            2462       0.71      0.65      0.68       284
            2522       0.90      0.88      0.89       998
            2582       0.75      0.66      0.70       518
            2583       0.92      0.97      0.95      2042
            2585       0.83      0.65      0.73       499
            2705       0.74      0.61      0.66       552
            2905       0.98      0.82      0.89       174
    
        accuracy                           0.76     16984
       macro avg       0.74      0.70      0.71     16984
    weighted avg       0.77      0.76      0.76     16984
    
    


```python
ds.show_confusion_matrix(y_orig, y_pred)
```


    
![png](images/ReadMe_ML_files/ReadMe_ML_82_0.png)
    



```python
df_cross = gboost.get_df_cross()
ds.Afficher_repartition(df_cross,Lcat,catdict)
```

    10    ------    livres
      : 10,  : 61.0 % , livres
      : 2280,  : 11.56 % , magazines
      : 2403,  : 10.43 % , livres et bds
      : 2705,  : 4.33 % , bds et livres
      : 40,  : 2.57 % , jeux video pour pc et consoles
    40    ------    jeux video pour pc et consoles
      : 40,  : 52.39 % , jeux video pour pc et consoles
      : 10,  : 11.95 % , livres
      : 50,  : 8.57 % ,  accesoires jeux video
      : 2462,  : 3.59 % , consoles de jeux video et jeux videos
      : 2280,  : 3.59 % , magazines
    50    ------     accesoires jeux video
      : 50,  : 62.5 % ,  accesoires jeux video
      : 10,  : 8.04 % , livres
      : 40,  : 4.76 % , jeux video pour pc et consoles
      : 2462,  : 4.46 % , consoles de jeux video et jeux videos
      : 1140,  : 2.38 % , produits derives “geeks” et figurines
    60    ------    consoles de jeux video
      : 60,  : 71.69 % , consoles de jeux video
      : 50,  : 9.64 % ,  accesoires jeux video
      : 1280,  : 5.42 % , jouets, peluches, poupees
      : 10,  : 4.22 % , livres
      : 2462,  : 4.22 % , consoles de jeux video et jeux videos
    1140    ------    produits derives “geeks” et figurines
      : 1140,  : 71.72 % , produits derives “geeks” et figurines
      : 1280,  : 4.49 % , jouets, peluches, poupees
      : 10,  : 3.93 % , livres
      : 2403,  : 3.0 % , livres et bds
      : 2280,  : 3.0 % , magazines
    1160    ------    cartes collectionables
      : 1160,  : 89.38 % , cartes collectionables
      : 10,  : 4.3 % , livres
      : 40,  : 1.14 % , jeux video pour pc et consoles
      : 1180,  : 1.14 % , figurines collectionables pour jeux de societe
      : 2280,  : 0.76 % , magazines
    1180    ------    figurines collectionables pour jeux de societe
      : 1180,  : 49.02 % , figurines collectionables pour jeux de societe
      : 10,  : 13.73 % , livres
      : 1140,  : 7.84 % , produits derives “geeks” et figurines
      : 1281,  : 5.23 % , jeux de societe/cartes
      : 1280,  : 5.23 % , jouets, peluches, poupees
    1280    ------    jouets, peluches, poupees
      : 1280,  : 63.24 % , jouets, peluches, poupees
      : 1300,  : 10.57 % , Petites voitures (jouets) et maquettes
      : 1281,  : 5.24 % , jeux de societe/cartes
      : 1140,  : 5.03 % , produits derives “geeks” et figurines
      : 10,  : 3.18 % , livres
    1281    ------    jeux de societe/cartes
      : 1281,  : 39.13 % , jeux de societe/cartes
      : 1280,  : 20.29 % , jouets, peluches, poupees
      : 10,  : 7.73 % , livres
      : 1160,  : 5.8 % , cartes collectionables
      : 2403,  : 5.31 % , livres et bds
    1300    ------    Petites voitures (jouets) et maquettes
      : 1300,  : 91.48 % , Petites voitures (jouets) et maquettes
      : 1280,  : 2.48 % , jouets, peluches, poupees
      : 10,  : 1.19 % , livres
      : 2280,  : 0.79 % , magazines
      : 2583,  : 0.5 % , accesoires de piscine
    1301    ------    accesoires pour petis enfants/bebes et mobilier de jeu (flechettes, billard, babyfoot)
      : 1301,  : 73.91 % , accesoires pour petis enfants/bebes et mobilier de jeu (flechettes, billard, babyfoot)
      : 10,  : 4.97 % , livres
      : 1280,  : 3.73 % , jouets, peluches, poupees
      : 2403,  : 2.48 % , livres et bds
      : 1320,  : 2.48 % , sacs pour femmes et accesore petite enfance
    1302    ------    jeux d'exterieur
      : 1302,  : 69.28 % , jeux d'exterieur
      : 1280,  : 9.24 % , jouets, peluches, poupees
      : 10,  : 3.61 % , livres
      : 2583,  : 2.81 % , accesoires de piscine
      : 2060,  : 1.81 % , lampes et accesoires decoration pour maison
    1320    ------    sacs pour femmes et accesore petite enfance
      : 1320,  : 67.59 % , sacs pour femmes et accesore petite enfance
      : 1280,  : 5.4 % , jouets, peluches, poupees
      : 10,  : 5.09 % , livres
      : 2060,  : 4.32 % , lampes et accesoires decoration pour maison
      : 1560,  : 3.24 % , Mobilier et produits decoration/rangement pour la maison
    1560    ------    Mobilier et produits decoration/rangement pour la maison
      : 1560,  : 79.51 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 6.01 % , lampes et accesoires decoration pour maison
      : 2582,  : 3.45 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 1920,  : 2.76 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 2583,  : 1.18 % , accesoires de piscine
    1920    ------    linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 1920,  : 88.85 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 1560,  : 3.83 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 2.32 % , lampes et accesoires decoration pour maison
      : 10,  : 1.05 % , livres
      : 1320,  : 0.7 % , sacs pour femmes et accesore petite enfance
    1940    ------    nouriture (cafes,infusions,conserves, epices,etc)
      : 1940,  : 56.52 % , nouriture (cafes,infusions,conserves, epices,etc)
      : 2403,  : 11.8 % , livres et bds
      : 10,  : 8.7 % , livres
      : 2583,  : 4.35 % , accesoires de piscine
      : 2280,  : 3.73 % , magazines
    2060    ------    lampes et accesoires decoration pour maison
      : 2060,  : 77.18 % , lampes et accesoires decoration pour maison
      : 1560,  : 5.51 % , Mobilier et produits decoration/rangement pour la maison
      : 1920,  : 2.9 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 10,  : 2.2 % , livres
      : 2582,  : 1.7 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
    2220    ------    accesoires mascots/pets
      : 2220,  : 42.42 % , accesoires mascots/pets
      : 2060,  : 10.91 % , lampes et accesoires decoration pour maison
      : 1560,  : 8.48 % , Mobilier et produits decoration/rangement pour la maison
      : 1280,  : 7.88 % , jouets, peluches, poupees
      : 10,  : 6.06 % , livres
    2280    ------    magazines
      : 2280,  : 76.68 % , magazines
      : 2403,  : 7.88 % , livres et bds
      : 10,  : 7.04 % , livres
      : 2705,  : 1.58 % , bds et livres
      : 1160,  : 1.16 % , cartes collectionables
    2403    ------    livres et bds
      : 2403,  : 72.25 % , livres et bds
      : 2280,  : 10.05 % , magazines
      : 10,  : 9.84 % , livres
      : 2705,  : 1.88 % , bds et livres
      : 1160,  : 0.94 % , cartes collectionables
    2462    ------    consoles de jeux video et jeux videos
      : 2462,  : 64.79 % , consoles de jeux video et jeux videos
      : 50,  : 7.39 % ,  accesoires jeux video
      : 10,  : 6.69 % , livres
      : 40,  : 5.63 % , jeux video pour pc et consoles
      : 2403,  : 4.58 % , livres et bds
    2522    ------    produits de papeterie et rangement bureau
      : 2522,  : 88.28 % , produits de papeterie et rangement bureau
      : 2403,  : 1.9 % , livres et bds
      : 10,  : 1.8 % , livres
      : 1560,  : 1.1 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 1.0 % , lampes et accesoires decoration pour maison
    2582    ------    mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 2582,  : 65.64 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 1560,  : 9.07 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 6.18 % , lampes et accesoires decoration pour maison
      : 2583,  : 5.21 % , accesoires de piscine
      : 2585,  : 3.09 % , outillage et accesoires pour jardinage
    2583    ------    accesoires de piscine
      : 2583,  : 97.45 % , accesoires de piscine
      : 10,  : 0.39 % , livres
      : 2060,  : 0.39 % , lampes et accesoires decoration pour maison
      : 1280,  : 0.24 % , jouets, peluches, poupees
      : 2403,  : 0.24 % , livres et bds
    2585    ------    outillage et accesoires pour jardinage
      : 2585,  : 65.33 % , outillage et accesoires pour jardinage
      : 2583,  : 8.02 % , accesoires de piscine
      : 10,  : 4.61 % , livres
      : 1560,  : 4.61 % , Mobilier et produits decoration/rangement pour la maison
      : 2582,  : 3.81 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
    2705    ------    bds et livres
      : 2705,  : 60.51 % , bds et livres
      : 10,  : 19.38 % , livres
      : 2280,  : 7.43 % , magazines
      : 2403,  : 5.8 % , livres et bds
      : 1281,  : 1.27 % , jeux de societe/cartes
    2905    ------    Jeu En téléchargement
      : 2905,  : 82.18 % , Jeu En téléchargement
      : 1280,  : 6.32 % , jouets, peluches, poupees
      : 2705,  : 4.6 % , bds et livres
      : 2403,  : 1.72 % , livres et bds
      : 1302,  : 0.57 % , jeux d'exterieur
    

    /content/Rakuten_Text_Classification_ML/Bibli_DataScience_3_1.py:170: FutureWarning: iteritems is deprecated and will be removed in a future version. Use .items instead.
      for index, value in s.iteritems():
    

### ***XGBClassifier***
la classe la classe **ML_XGBClassifier** utilise un pipeline :


```
text_xgboost = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),
            ('clf', xgb.XGBClassifier(learning_rate=0.1,n_estimators=500,max_depth=10)),
            ])
```



```python
xgboost= ml.ML_XGBClassifier("XGBClassifier")
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    

    preprocessing ...
    


```python
xgboost.fit_modele(savefics=True,Train="Load")
```

    L'heure au début de l'entraînement était :  2024-01-29 17:43:47.991567
    L'heure à la fin de l'entraînement était :  2024-01-29 22:13:54.936336
    F1 Score:  0.8223380676791777
    Accuracy:  0.8178285445124823
    


```python
y_orig = xgboost.get_y_orig()
y_pred = xgboost.get_y_pred()
_,_ = ds.get_classification_report(y_orig, y_pred)
```

    Précision de la prédiction: 81.78285445124823 %
    Evaluation détaillée de la Classification par RDF :
     
                   precision    recall  f1-score   support
    
              10       0.37      0.74      0.50       623
              40       0.78      0.62      0.69       502
              50       0.76      0.78      0.77       336
              60       0.95      0.81      0.88       166
            1140       0.81      0.76      0.78       534
            1160       0.91      0.90      0.91       791
            1180       0.79      0.60      0.68       153
            1280       0.75      0.75      0.75       974
            1281       0.67      0.57      0.62       414
            1300       0.95      0.94      0.95      1009
            1301       0.94      0.88      0.91       161
            1302       0.88      0.80      0.84       498
            1320       0.84      0.77      0.80       648
            1560       0.85      0.83      0.84      1015
            1920       0.91      0.93      0.92       861
            1940       0.89      0.81      0.85       161
            2060       0.82      0.83      0.83       999
            2220       0.83      0.74      0.78       165
            2280       0.76      0.79      0.77       952
            2403       0.77      0.74      0.75       955
            2462       0.75      0.73      0.74       284
            2522       0.91      0.90      0.91       998
            2582       0.80      0.73      0.76       518
            2583       0.96      0.97      0.97      2042
            2585       0.82      0.75      0.78       499
            2705       0.82      0.66      0.73       552
            2905       1.00      0.99      1.00       174
    
        accuracy                           0.82     16984
       macro avg       0.83      0.79      0.80     16984
    weighted avg       0.83      0.82      0.82     16984
    
    


```python
ds.show_confusion_matrix(y_orig, y_pred)
```


    

![png](images/ReadMe_ML_files/ReadMe_ML_90_0.png)
    



```python
df_cross =xgboost.get_df_cross()
Lcat=xgboost.get_cat()
catdict = xgboost.get_catdict()
ds.Afficher_repartition(df_cross,Lcat,catdict)
```

    10    ------    livres
      : 10,  : 73.68 % , livres
      : 2280,  : 9.15 % , magazines
      : 2403,  : 5.78 % , livres et bds
      : 2705,  : 4.98 % , bds et livres
      : 40,  : 1.61 % , jeux video pour pc et consoles
    40    ------    jeux video pour pc et consoles
      : 40,  : 61.95 % , jeux video pour pc et consoles
      : 10,  : 14.34 % , livres
      : 50,  : 4.98 % ,  accesoires jeux video
      : 2462,  : 3.78 % , consoles de jeux video et jeux videos
      : 1280,  : 2.39 % , jouets, peluches, poupees
    50    ------     accesoires jeux video
      : 50,  : 77.98 % ,  accesoires jeux video
      : 2462,  : 7.44 % , consoles de jeux video et jeux videos
      : 40,  : 2.98 % , jeux video pour pc et consoles
      : 10,  : 2.38 % , livres
      : 1140,  : 2.38 % , produits derives “geeks” et figurines
    60    ------    consoles de jeux video
      : 60,  : 81.33 % , consoles de jeux video
      : 50,  : 9.04 % ,  accesoires jeux video
      : 2462,  : 5.42 % , consoles de jeux video et jeux videos
      : 40,  : 1.81 % , jeux video pour pc et consoles
      : 10,  : 0.6 % , livres
    1140    ------    produits derives “geeks” et figurines
      : 1140,  : 76.22 % , produits derives “geeks” et figurines
      : 10,  : 4.68 % , livres
      : 1280,  : 4.12 % , jouets, peluches, poupees
      : 2280,  : 2.25 % , magazines
      : 1180,  : 2.06 % , figurines collectionables pour jeux de societe
    1160    ------    cartes collectionables
      : 1160,  : 90.27 % , cartes collectionables
      : 10,  : 5.06 % , livres
      : 40,  : 1.39 % , jeux video pour pc et consoles
      : 2280,  : 1.26 % , magazines
      : 2403,  : 0.76 % , livres et bds
    1180    ------    figurines collectionables pour jeux de societe
      : 1180,  : 60.13 % , figurines collectionables pour jeux de societe
      : 10,  : 13.73 % , livres
      : 1140,  : 3.92 % , produits derives “geeks” et figurines
      : 2403,  : 3.92 % , livres et bds
      : 40,  : 3.27 % , jeux video pour pc et consoles
    1280    ------    jouets, peluches, poupees
      : 1280,  : 75.26 % , jouets, peluches, poupees
      : 1281,  : 5.13 % , jeux de societe/cartes
      : 1140,  : 4.93 % , produits derives “geeks” et figurines
      : 10,  : 3.08 % , livres
      : 1300,  : 1.64 % , Petites voitures (jouets) et maquettes
    1281    ------    jeux de societe/cartes
      : 1281,  : 57.0 % , jeux de societe/cartes
      : 1280,  : 17.63 % , jouets, peluches, poupees
      : 10,  : 6.28 % , livres
      : 1160,  : 3.14 % , cartes collectionables
      : 2403,  : 2.66 % , livres et bds
    1300    ------    Petites voitures (jouets) et maquettes
      : 1300,  : 94.05 % , Petites voitures (jouets) et maquettes
      : 1280,  : 2.08 % , jouets, peluches, poupees
      : 10,  : 1.78 % , livres
      : 2403,  : 0.4 % , livres et bds
      : 2583,  : 0.4 % , accesoires de piscine
    1301    ------    accesoires pour petis enfants/bebes et mobilier de jeu (flechettes, billard, babyfoot)
      : 1301,  : 88.2 % , accesoires pour petis enfants/bebes et mobilier de jeu (flechettes, billard, babyfoot)
      : 10,  : 2.48 % , livres
      : 2522,  : 1.86 % , produits de papeterie et rangement bureau
      : 1280,  : 1.86 % , jouets, peluches, poupees
      : 2583,  : 1.24 % , accesoires de piscine
    1302    ------    jeux d'exterieur
      : 1302,  : 79.92 % , jeux d'exterieur
      : 1280,  : 5.42 % , jouets, peluches, poupees
      : 10,  : 2.61 % , livres
      : 1281,  : 2.61 % , jeux de societe/cartes
      : 2583,  : 1.61 % , accesoires de piscine
    1320    ------    sacs pour femmes et accesore petite enfance
      : 1320,  : 76.7 % , sacs pour femmes et accesore petite enfance
      : 10,  : 4.48 % , livres
      : 1280,  : 3.55 % , jouets, peluches, poupees
      : 2060,  : 2.93 % , lampes et accesoires decoration pour maison
      : 1920,  : 2.62 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
    1560    ------    Mobilier et produits decoration/rangement pour la maison
      : 1560,  : 82.76 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 4.63 % , lampes et accesoires decoration pour maison
      : 2582,  : 3.25 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 1920,  : 2.56 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 2585,  : 1.77 % , outillage et accesoires pour jardinage
    1920    ------    linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 1920,  : 93.38 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 1560,  : 2.21 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 2.21 % , lampes et accesoires decoration pour maison
      : 1320,  : 0.7 % , sacs pour femmes et accesore petite enfance
      : 1280,  : 0.46 % , jouets, peluches, poupees
    1940    ------    nouriture (cafes,infusions,conserves, epices,etc)
      : 1940,  : 81.37 % , nouriture (cafes,infusions,conserves, epices,etc)
      : 10,  : 6.83 % , livres
      : 1280,  : 1.86 % , jouets, peluches, poupees
      : 2403,  : 1.86 % , livres et bds
      : 2705,  : 1.24 % , bds et livres
    2060    ------    lampes et accesoires decoration pour maison
      : 2060,  : 83.18 % , lampes et accesoires decoration pour maison
      : 1560,  : 3.8 % , Mobilier et produits decoration/rangement pour la maison
      : 1920,  : 2.4 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 2582,  : 1.8 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 10,  : 1.4 % , livres
    2220    ------    accesoires mascots/pets
      : 2220,  : 73.94 % , accesoires mascots/pets
      : 2522,  : 3.64 % , produits de papeterie et rangement bureau
      : 1320,  : 3.64 % , sacs pour femmes et accesore petite enfance
      : 2060,  : 3.03 % , lampes et accesoires decoration pour maison
      : 1280,  : 2.42 % , jouets, peluches, poupees
    2280    ------    magazines
      : 2280,  : 79.31 % , magazines
      : 10,  : 11.55 % , livres
      : 2403,  : 5.15 % , livres et bds
      : 1160,  : 1.05 % , cartes collectionables
      : 2705,  : 1.05 % , bds et livres
    2403    ------    livres et bds
      : 2403,  : 73.82 % , livres et bds
      : 10,  : 14.24 % , livres
      : 2280,  : 7.54 % , magazines
      : 2705,  : 1.05 % , bds et livres
      : 1160,  : 0.84 % , cartes collectionables
    2462    ------    consoles de jeux video et jeux videos
      : 2462,  : 72.89 % , consoles de jeux video et jeux videos
      : 50,  : 9.51 % ,  accesoires jeux video
      : 40,  : 5.63 % , jeux video pour pc et consoles
      : 10,  : 2.82 % , livres
      : 2403,  : 2.11 % , livres et bds
    2522    ------    produits de papeterie et rangement bureau
      : 2522,  : 89.68 % , produits de papeterie et rangement bureau
      : 10,  : 2.3 % , livres
      : 2403,  : 1.6 % , livres et bds
      : 1560,  : 1.3 % , Mobilier et produits decoration/rangement pour la maison
      : 2280,  : 1.0 % , magazines
    2582    ------    mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 2582,  : 72.78 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 1560,  : 6.37 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 5.79 % , lampes et accesoires decoration pour maison
      : 2585,  : 4.05 % , outillage et accesoires pour jardinage
      : 2583,  : 2.7 % , accesoires de piscine
    2583    ------    accesoires de piscine
      : 2583,  : 96.77 % , accesoires de piscine
      : 10,  : 0.59 % , livres
      : 2060,  : 0.39 % , lampes et accesoires decoration pour maison
      : 2585,  : 0.34 % , outillage et accesoires pour jardinage
      : 1302,  : 0.24 % , jeux d'exterieur
    2585    ------    outillage et accesoires pour jardinage
      : 2585,  : 75.35 % , outillage et accesoires pour jardinage
      : 2583,  : 4.01 % , accesoires de piscine
      : 1560,  : 3.81 % , Mobilier et produits decoration/rangement pour la maison
      : 2582,  : 3.61 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 2060,  : 3.01 % , lampes et accesoires decoration pour maison
    2705    ------    bds et livres
      : 2705,  : 65.76 % , bds et livres
      : 10,  : 25.0 % , livres
      : 2403,  : 3.44 % , livres et bds
      : 2280,  : 2.72 % , magazines
      : 1320,  : 0.72 % , sacs pour femmes et accesore petite enfance
    2905    ------    Jeu En téléchargement
      : 2905,  : 99.43 % , Jeu En téléchargement
      : 1281,  : 0.57 % , jeux de societe/cartes
      : 1920,  : 0.0 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 2705,  : 0.0 % , bds et livres
      : 2585,  : 0.0 % , outillage et accesoires pour jardinage
    

    /content/Rakuten_Text_Classification_ML/Bibli_DataScience_3_1.py:170: FutureWarning: iteritems is deprecated and will be removed in a future version. Use .items instead.
      for index, value in s.iteritems():
    

### ***MultinomialNB***
la classe la classe **ML_MultinomialNB** utilise un pipeline :

```
text_NB = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),
            ('clf', MultinomialNB(alpha = 0.1, fit_prior = False)),
            ])
```



```python
NB = ml.ML_MultinomialNB("MultinomialNB")
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    

    preprocessing ...
    


```python
NB.fit_modele(savefics=True,Train="Load")
```

    L'heure au début de l'entraînement était :  2024-01-28 20:04:15.479809
    L'heure à la fin de l'entraînement était :  2024-01-28 20:04:21.349130
    F1 Score:  0.7869377756174585
    Accuracy:  0.7910386245878473
    


```python
y_orig = NB.get_y_orig()
y_pred = NB.get_y_pred()
_,_ = ds.get_classification_report(y_orig, y_pred)
```

    Précision de la prédiction: 79.10386245878473 %
    Evaluation détaillée de la Classification par RDF :
     
                   precision    recall  f1-score   support
    
              10       0.68      0.40      0.50       623
              40       0.76      0.56      0.65       502
              50       0.69      0.79      0.73       336
              60       0.86      0.74      0.80       166
            1140       0.66      0.82      0.73       534
            1160       0.93      0.96      0.95       791
            1180       0.73      0.61      0.66       153
            1280       0.65      0.56      0.60       974
            1281       0.67      0.49      0.56       414
            1300       0.81      0.94      0.87      1009
            1301       0.91      0.91      0.91       161
            1302       0.76      0.76      0.76       498
            1320       0.82      0.75      0.79       648
            1560       0.75      0.75      0.75      1015
            1920       0.88      0.90      0.89       861
            1940       0.82      0.91      0.86       161
            2060       0.70      0.79      0.74       999
            2220       0.80      0.75      0.77       165
            2280       0.74      0.80      0.77       952
            2403       0.74      0.75      0.75       955
            2462       0.66      0.74      0.69       284
            2522       0.94      0.88      0.91       998
            2582       0.73      0.72      0.73       518
            2583       0.96      0.96      0.96      2042
            2585       0.77      0.81      0.79       499
            2705       0.70      0.77      0.74       552
            2905       0.70      0.99      0.82       174
    
        accuracy                           0.79     16984
       macro avg       0.77      0.77      0.77     16984
    weighted avg       0.79      0.79      0.79     16984
    
    


```python
ds.show_confusion_matrix(y_orig, y_pred)
```


    
![png](images/ReadMe_ML_files/ReadMe_ML_95_0.png)
    



```python
df_cross =NB.get_df_cross()
ds.Afficher_repartition(df_cross,Lcat,catdict)
```

    10    ------    livres
      : 10,  : 39.81 % , livres
      : 2280,  : 13.64 % , magazines
      : 2705,  : 13.48 % , bds et livres
      : 2403,  : 11.88 % , livres et bds
      : 40,  : 2.25 % , jeux video pour pc et consoles
    40    ------    jeux video pour pc et consoles
      : 40,  : 56.37 % , jeux video pour pc et consoles
      : 50,  : 10.16 % ,  accesoires jeux video
      : 2462,  : 6.37 % , consoles de jeux video et jeux videos
      : 1140,  : 4.78 % , produits derives “geeks” et figurines
      : 2905,  : 4.38 % , Jeu En téléchargement
    50    ------     accesoires jeux video
      : 50,  : 78.57 % ,  accesoires jeux video
      : 2462,  : 8.93 % , consoles de jeux video et jeux videos
      : 60,  : 2.68 % , consoles de jeux video
      : 1140,  : 2.38 % , produits derives “geeks” et figurines
      : 40,  : 1.79 % , jeux video pour pc et consoles
    60    ------    consoles de jeux video
      : 60,  : 74.1 % , consoles de jeux video
      : 2462,  : 13.25 % , consoles de jeux video et jeux videos
      : 50,  : 7.23 % ,  accesoires jeux video
      : 40,  : 3.01 % , jeux video pour pc et consoles
      : 2905,  : 1.2 % , Jeu En téléchargement
    1140    ------    produits derives “geeks” et figurines
      : 1140,  : 82.4 % , produits derives “geeks” et figurines
      : 1280,  : 4.31 % , jouets, peluches, poupees
      : 40,  : 1.69 % , jeux video pour pc et consoles
      : 2280,  : 1.69 % , magazines
      : 2705,  : 1.5 % , bds et livres
    1160    ------    cartes collectionables
      : 1160,  : 95.95 % , cartes collectionables
      : 1140,  : 1.39 % , produits derives “geeks” et figurines
      : 1281,  : 0.76 % , jeux de societe/cartes
      : 40,  : 0.38 % , jeux video pour pc et consoles
      : 2280,  : 0.38 % , magazines
    1180    ------    figurines collectionables pour jeux de societe
      : 1180,  : 60.78 % , figurines collectionables pour jeux de societe
      : 1140,  : 6.54 % , produits derives “geeks” et figurines
      : 1281,  : 5.88 % , jeux de societe/cartes
      : 1280,  : 5.23 % , jouets, peluches, poupees
      : 2905,  : 3.92 % , Jeu En téléchargement
    1280    ------    jouets, peluches, poupees
      : 1280,  : 55.75 % , jouets, peluches, poupees
      : 1300,  : 18.48 % , Petites voitures (jouets) et maquettes
      : 1140,  : 7.91 % , produits derives “geeks” et figurines
      : 1281,  : 3.7 % , jeux de societe/cartes
      : 1302,  : 3.49 % , jeux d'exterieur
    1281    ------    jeux de societe/cartes
      : 1281,  : 48.79 % , jeux de societe/cartes
      : 1280,  : 18.36 % , jouets, peluches, poupees
      : 2905,  : 3.86 % , Jeu En téléchargement
      : 1160,  : 3.14 % , cartes collectionables
      : 2280,  : 3.14 % , magazines
    1300    ------    Petites voitures (jouets) et maquettes
      : 1300,  : 94.45 % , Petites voitures (jouets) et maquettes
      : 1280,  : 2.28 % , jouets, peluches, poupees
      : 2280,  : 0.69 % , magazines
      : 2585,  : 0.4 % , outillage et accesoires pour jardinage
      : 50,  : 0.4 % ,  accesoires jeux video
    1301    ------    accesoires pour petis enfants/bebes et mobilier de jeu (flechettes, billard, babyfoot)
      : 1301,  : 90.68 % , accesoires pour petis enfants/bebes et mobilier de jeu (flechettes, billard, babyfoot)
      : 1280,  : 3.11 % , jouets, peluches, poupees
      : 1560,  : 1.24 % , Mobilier et produits decoration/rangement pour la maison
      : 2582,  : 1.24 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 1302,  : 1.24 % , jeux d'exterieur
    1302    ------    jeux d'exterieur
      : 1302,  : 75.5 % , jeux d'exterieur
      : 1280,  : 7.63 % , jouets, peluches, poupees
      : 1320,  : 2.41 % , sacs pour femmes et accesore petite enfance
      : 2583,  : 2.01 % , accesoires de piscine
      : 1560,  : 1.81 % , Mobilier et produits decoration/rangement pour la maison
    1320    ------    sacs pour femmes et accesore petite enfance
      : 1320,  : 75.31 % , sacs pour femmes et accesore petite enfance
      : 2060,  : 7.41 % , lampes et accesoires decoration pour maison
      : 1280,  : 5.4 % , jouets, peluches, poupees
      : 1560,  : 3.7 % , Mobilier et produits decoration/rangement pour la maison
      : 1920,  : 2.01 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
    1560    ------    Mobilier et produits decoration/rangement pour la maison
      : 1560,  : 75.47 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 9.95 % , lampes et accesoires decoration pour maison
      : 2582,  : 5.52 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 1920,  : 3.74 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 2585,  : 2.36 % , outillage et accesoires pour jardinage
    1920    ------    linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 1920,  : 90.24 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 1560,  : 3.83 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 3.25 % , lampes et accesoires decoration pour maison
      : 1320,  : 1.28 % , sacs pour femmes et accesore petite enfance
      : 2582,  : 0.46 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
    1940    ------    nouriture (cafes,infusions,conserves, epices,etc)
      : 1940,  : 90.68 % , nouriture (cafes,infusions,conserves, epices,etc)
      : 1560,  : 3.11 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 1.24 % , lampes et accesoires decoration pour maison
      : 1280,  : 1.24 % , jouets, peluches, poupees
      : 1320,  : 0.62 % , sacs pour femmes et accesore petite enfance
    2060    ------    lampes et accesoires decoration pour maison
      : 2060,  : 79.48 % , lampes et accesoires decoration pour maison
      : 1560,  : 7.11 % , Mobilier et produits decoration/rangement pour la maison
      : 1920,  : 4.3 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 2582,  : 1.7 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 2585,  : 1.1 % , outillage et accesoires pour jardinage
    2220    ------    accesoires mascots/pets
      : 2220,  : 74.55 % , accesoires mascots/pets
      : 2060,  : 5.45 % , lampes et accesoires decoration pour maison
      : 1320,  : 3.64 % , sacs pour femmes et accesore petite enfance
      : 1280,  : 3.03 % , jouets, peluches, poupees
      : 2585,  : 2.42 % , outillage et accesoires pour jardinage
    2280    ------    magazines
      : 2280,  : 79.73 % , magazines
      : 2403,  : 9.24 % , livres et bds
      : 2705,  : 2.52 % , bds et livres
      : 1140,  : 2.1 % , produits derives “geeks” et figurines
      : 10,  : 1.58 % , livres
    2403    ------    livres et bds
      : 2403,  : 74.97 % , livres et bds
      : 2280,  : 10.99 % , magazines
      : 10,  : 4.29 % , livres
      : 2705,  : 3.77 % , bds et livres
      : 1140,  : 1.68 % , produits derives “geeks” et figurines
    2462    ------    consoles de jeux video et jeux videos
      : 2462,  : 73.59 % , consoles de jeux video et jeux videos
      : 50,  : 9.86 % ,  accesoires jeux video
      : 40,  : 8.1 % , jeux video pour pc et consoles
      : 2905,  : 1.41 % , Jeu En téléchargement
      : 1140,  : 1.41 % , produits derives “geeks” et figurines
    2522    ------    produits de papeterie et rangement bureau
      : 2522,  : 88.18 % , produits de papeterie et rangement bureau
      : 1560,  : 2.1 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 2.1 % , lampes et accesoires decoration pour maison
      : 2585,  : 1.6 % , outillage et accesoires pour jardinage
      : 2403,  : 1.4 % , livres et bds
    2582    ------    mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 2582,  : 72.39 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 1560,  : 8.69 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 7.14 % , lampes et accesoires decoration pour maison
      : 2585,  : 5.02 % , outillage et accesoires pour jardinage
      : 2583,  : 1.93 % , accesoires de piscine
    2583    ------    accesoires de piscine
      : 2583,  : 96.43 % , accesoires de piscine
      : 1302,  : 0.93 % , jeux d'exterieur
      : 2060,  : 0.93 % , lampes et accesoires decoration pour maison
      : 2585,  : 0.44 % , outillage et accesoires pour jardinage
      : 2582,  : 0.34 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
    2585    ------    outillage et accesoires pour jardinage
      : 2585,  : 81.36 % , outillage et accesoires pour jardinage
      : 2060,  : 4.61 % , lampes et accesoires decoration pour maison
      : 2583,  : 3.81 % , accesoires de piscine
      : 1560,  : 3.61 % , Mobilier et produits decoration/rangement pour la maison
      : 2582,  : 2.4 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
    2705    ------    bds et livres
      : 2705,  : 76.99 % , bds et livres
      : 10,  : 5.8 % , livres
      : 2403,  : 4.53 % , livres et bds
      : 2280,  : 3.8 % , magazines
      : 2220,  : 0.91 % , accesoires mascots/pets
    2905    ------    Jeu En téléchargement
      : 2905,  : 98.85 % , Jeu En téléchargement
      : 2705,  : 1.15 % , bds et livres
      : 1920,  : 0.0 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 2585,  : 0.0 % , outillage et accesoires pour jardinage
      : 2583,  : 0.0 % , accesoires de piscine
    

    /content/Rakuten_Text_Classification_ML/Bibli_DataScience_3_1.py:170: FutureWarning: iteritems is deprecated and will be removed in a future version. Use .items instead.
      for index, value in s.iteritems():
    

### ***DecisionTreeClassifier***
la classe la classe **ML_DecisionTreeClassifier** utilise un pipeline :

```
text_DTCL = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True,max_df=0.8, min_df=2)),
            ('clf', DecisionTreeClassifier(class_weight='balanced')),
            ])
```


```python
DTCL = ml.ML_DecisionTreeClassifier("DecisionTreeClassifier")
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    

    preprocessing ...
    


```python
DTCL.fit_modele(savefics=True,Train="Load")
```

    L'heure au début de l'entraînement était :  2024-01-28 20:10:59.124348
    L'heure à la fin de l'entraînement était :  2024-01-28 20:12:54.420571
    F1 Score:  0.728998983854548
    Accuracy:  0.7268016957136129
    


```python
y_orig = DTCL.get_y_orig()
y_pred = DTCL.get_y_pred()
_,_ = ds.get_classification_report(y_orig, y_pred)
```

    Précision de la prédiction: 72.68016957136129 %
    Evaluation détaillée de la Classification par RDF :
     
                   precision    recall  f1-score   support
    
              10       0.38      0.61      0.47       623
              40       0.65      0.56      0.60       502
              50       0.66      0.68      0.67       336
              60       0.85      0.83      0.84       166
            1140       0.71      0.69      0.70       534
            1160       0.88      0.87      0.88       791
            1180       0.64      0.59      0.61       153
            1280       0.63      0.57      0.60       974
            1281       0.51      0.50      0.51       414
            1300       0.92      0.91      0.92      1009
            1301       0.91      0.85      0.88       161
            1302       0.74      0.66      0.70       498
            1320       0.64      0.62      0.63       648
            1560       0.69      0.68      0.68      1015
            1920       0.87      0.85      0.86       861
            1940       0.68      0.71      0.69       161
            2060       0.68      0.65      0.67       999
            2220       0.69      0.63      0.66       165
            2280       0.70      0.72      0.71       952
            2403       0.70      0.69      0.70       955
            2462       0.66      0.67      0.66       284
            2522       0.76      0.79      0.78       998
            2582       0.62      0.60      0.61       518
            2583       0.93      0.91      0.92      2042
            2585       0.63      0.64      0.64       499
            2705       0.62      0.60      0.61       552
            2905       0.99      0.98      0.99       174
    
        accuracy                           0.73     16984
       macro avg       0.72      0.71      0.71     16984
    weighted avg       0.73      0.73      0.73     16984
    
    


```python
ds.show_confusion_matrix(y_orig, y_pred)
```


    
![png](images/ReadMe_ML_files/ReadMe_ML_101_0.png)
    



```python
df_cross =DTCL.get_df_cross()
ds.Afficher_repartition(df_cross,Lcat,catdict)
```

    10    ------    livres
      : 10,  : 60.67 % , livres
      : 2280,  : 9.79 % , magazines
      : 2705,  : 8.51 % , bds et livres
      : 2403,  : 7.38 % , livres et bds
      : 40,  : 4.49 % , jeux video pour pc et consoles
    40    ------    jeux video pour pc et consoles
      : 40,  : 55.78 % , jeux video pour pc et consoles
      : 10,  : 9.36 % , livres
      : 50,  : 6.57 % ,  accesoires jeux video
      : 2462,  : 4.98 % , consoles de jeux video et jeux videos
      : 1281,  : 3.59 % , jeux de societe/cartes
    50    ------     accesoires jeux video
      : 50,  : 67.56 % ,  accesoires jeux video
      : 2462,  : 7.74 % , consoles de jeux video et jeux videos
      : 40,  : 3.57 % , jeux video pour pc et consoles
      : 2522,  : 3.57 % , produits de papeterie et rangement bureau
      : 1280,  : 2.38 % , jouets, peluches, poupees
    60    ------    consoles de jeux video
      : 60,  : 83.13 % , consoles de jeux video
      : 50,  : 8.43 % ,  accesoires jeux video
      : 2462,  : 3.61 % , consoles de jeux video et jeux videos
      : 10,  : 1.2 % , livres
      : 1300,  : 0.6 % , Petites voitures (jouets) et maquettes
    1140    ------    produits derives “geeks” et figurines
      : 1140,  : 69.48 % , produits derives “geeks” et figurines
      : 1280,  : 5.43 % , jouets, peluches, poupees
      : 10,  : 2.81 % , livres
      : 2280,  : 2.62 % , magazines
      : 2403,  : 2.43 % , livres et bds
    1160    ------    cartes collectionables
      : 1160,  : 87.48 % , cartes collectionables
      : 10,  : 4.17 % , livres
      : 2280,  : 1.64 % , magazines
      : 40,  : 1.52 % , jeux video pour pc et consoles
      : 1281,  : 1.26 % , jeux de societe/cartes
    1180    ------    figurines collectionables pour jeux de societe
      : 1180,  : 58.82 % , figurines collectionables pour jeux de societe
      : 10,  : 12.42 % , livres
      : 1140,  : 5.23 % , produits derives “geeks” et figurines
      : 2280,  : 3.92 % , magazines
      : 1281,  : 3.92 % , jeux de societe/cartes
    1280    ------    jouets, peluches, poupees
      : 1280,  : 57.19 % , jouets, peluches, poupees
      : 1281,  : 8.42 % , jeux de societe/cartes
      : 1140,  : 4.83 % , produits derives “geeks” et figurines
      : 1320,  : 3.7 % , sacs pour femmes et accesore petite enfance
      : 10,  : 3.29 % , livres
    1281    ------    jeux de societe/cartes
      : 1281,  : 50.48 % , jeux de societe/cartes
      : 1280,  : 13.77 % , jouets, peluches, poupees
      : 1160,  : 4.59 % , cartes collectionables
      : 10,  : 4.11 % , livres
      : 40,  : 3.14 % , jeux video pour pc et consoles
    1300    ------    Petites voitures (jouets) et maquettes
      : 1300,  : 91.48 % , Petites voitures (jouets) et maquettes
      : 1280,  : 2.38 % , jouets, peluches, poupees
      : 10,  : 1.39 % , livres
      : 2060,  : 0.69 % , lampes et accesoires decoration pour maison
      : 2280,  : 0.59 % , magazines
    1301    ------    accesoires pour petis enfants/bebes et mobilier de jeu (flechettes, billard, babyfoot)
      : 1301,  : 85.09 % , accesoires pour petis enfants/bebes et mobilier de jeu (flechettes, billard, babyfoot)
      : 2522,  : 2.48 % , produits de papeterie et rangement bureau
      : 2583,  : 1.86 % , accesoires de piscine
      : 1281,  : 1.86 % , jeux de societe/cartes
      : 10,  : 1.86 % , livres
    1302    ------    jeux d'exterieur
      : 1302,  : 66.06 % , jeux d'exterieur
      : 1280,  : 8.03 % , jouets, peluches, poupees
      : 1281,  : 3.21 % , jeux de societe/cartes
      : 1320,  : 3.01 % , sacs pour femmes et accesore petite enfance
      : 1300,  : 2.21 % , Petites voitures (jouets) et maquettes
    1320    ------    sacs pour femmes et accesore petite enfance
      : 1320,  : 62.35 % , sacs pour femmes et accesore petite enfance
      : 1280,  : 4.78 % , jouets, peluches, poupees
      : 2060,  : 4.17 % , lampes et accesoires decoration pour maison
      : 2522,  : 3.55 % , produits de papeterie et rangement bureau
      : 1302,  : 3.24 % , jeux d'exterieur
    1560    ------    Mobilier et produits decoration/rangement pour la maison
      : 1560,  : 67.59 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 9.06 % , lampes et accesoires decoration pour maison
      : 2582,  : 5.32 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 1920,  : 3.25 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 2522,  : 2.96 % , produits de papeterie et rangement bureau
    1920    ------    linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 1920,  : 84.79 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
      : 1560,  : 3.6 % , Mobilier et produits decoration/rangement pour la maison
      : 1320,  : 2.67 % , sacs pour femmes et accesore petite enfance
      : 2060,  : 2.67 % , lampes et accesoires decoration pour maison
      : 2522,  : 1.16 % , produits de papeterie et rangement bureau
    1940    ------    nouriture (cafes,infusions,conserves, epices,etc)
      : 1940,  : 70.81 % , nouriture (cafes,infusions,conserves, epices,etc)
      : 10,  : 5.59 % , livres
      : 2522,  : 5.59 % , produits de papeterie et rangement bureau
      : 1320,  : 3.11 % , sacs pour femmes et accesore petite enfance
      : 2403,  : 3.11 % , livres et bds
    2060    ------    lampes et accesoires decoration pour maison
      : 2060,  : 65.27 % , lampes et accesoires decoration pour maison
      : 1560,  : 8.11 % , Mobilier et produits decoration/rangement pour la maison
      : 2582,  : 4.1 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 2585,  : 3.2 % , outillage et accesoires pour jardinage
      : 1920,  : 2.9 % , linge de maison (cousins, rideaux, serviettes, nappes, draps)
    2220    ------    accesoires mascots/pets
      : 2220,  : 63.03 % , accesoires mascots/pets
      : 2280,  : 4.24 % , magazines
      : 1320,  : 4.24 % , sacs pour femmes et accesore petite enfance
      : 1560,  : 3.64 % , Mobilier et produits decoration/rangement pour la maison
      : 2585,  : 3.03 % , outillage et accesoires pour jardinage
    2280    ------    magazines
      : 2280,  : 71.95 % , magazines
      : 10,  : 10.5 % , livres
      : 2403,  : 6.3 % , livres et bds
      : 2705,  : 3.26 % , bds et livres
      : 1160,  : 1.68 % , cartes collectionables
    2403    ------    livres et bds
      : 2403,  : 69.21 % , livres et bds
      : 10,  : 12.36 % , livres
      : 2280,  : 8.17 % , magazines
      : 2705,  : 2.83 % , bds et livres
      : 2522,  : 1.88 % , produits de papeterie et rangement bureau
    2462    ------    consoles de jeux video et jeux videos
      : 2462,  : 66.55 % , consoles de jeux video et jeux videos
      : 50,  : 7.75 % ,  accesoires jeux video
      : 40,  : 6.69 % , jeux video pour pc et consoles
      : 2403,  : 3.87 % , livres et bds
      : 60,  : 3.17 % , consoles de jeux video
    2522    ------    produits de papeterie et rangement bureau
      : 2522,  : 79.46 % , produits de papeterie et rangement bureau
      : 1320,  : 2.51 % , sacs pour femmes et accesore petite enfance
      : 2403,  : 2.1 % , livres et bds
      : 1560,  : 2.0 % , Mobilier et produits decoration/rangement pour la maison
      : 2585,  : 1.8 % , outillage et accesoires pour jardinage
    2582    ------    mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 2582,  : 60.23 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 1560,  : 11.0 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 6.95 % , lampes et accesoires decoration pour maison
      : 2585,  : 4.63 % , outillage et accesoires pour jardinage
      : 2522,  : 3.47 % , produits de papeterie et rangement bureau
    2583    ------    accesoires de piscine
      : 2583,  : 90.99 % , accesoires de piscine
      : 2585,  : 1.62 % , outillage et accesoires pour jardinage
      : 1302,  : 0.78 % , jeux d'exterieur
      : 1560,  : 0.64 % , Mobilier et produits decoration/rangement pour la maison
      : 1320,  : 0.64 % , sacs pour femmes et accesore petite enfance
    2585    ------    outillage et accesoires pour jardinage
      : 2585,  : 64.33 % , outillage et accesoires pour jardinage
      : 2582,  : 5.61 % , mobilier d'exterieur et accesoires (parasols,pots,tentes,etc)
      : 1560,  : 5.01 % , Mobilier et produits decoration/rangement pour la maison
      : 2060,  : 4.81 % , lampes et accesoires decoration pour maison
      : 2583,  : 4.21 % , accesoires de piscine
    2705    ------    bds et livres
      : 2705,  : 60.33 % , bds et livres
      : 10,  : 20.65 % , livres
      : 2403,  : 5.8 % , livres et bds
      : 2280,  : 4.35 % , magazines
      : 1320,  : 1.81 % , sacs pour femmes et accesore petite enfance
    2905    ------    Jeu En téléchargement
      : 2905,  : 98.28 % , Jeu En téléchargement
      : 50,  : 0.57 % ,  accesoires jeux video
      : 60,  : 0.57 % , consoles de jeux video
      : 1180,  : 0.57 % , figurines collectionables pour jeux de societe
      : 1940,  : 0.0 % , nouriture (cafes,infusions,conserves, epices,etc)
    

    /content/Rakuten_Text_Classification_ML/Bibli_DataScience_3_1.py:170: FutureWarning: iteritems is deprecated and will be removed in a future version. Use .items instead.
      for index, value in s.iteritems():
    

***Résultats***

| Modèle 		| F1 Score | Accuracy 	| Durée d'entrainement |
|---     		|:-:       |:-:       	| ---                  |  
|SVC	 		|0.8294    |0.8256	|3:21:42|	
|LogisticRegression	|0.8018    |0.8022	|0:10:00|
|RandomForestClassifier	|0.7933	   |0.7926	|0:08:22|
|XGBClassifier		|0.8223	   |0.8178	|4:30:17|
|MultinomialNB		|0.7869	   |0.7910	|0:00:06|
|DecisionTreeClassifier |0.7289	   |0.7268	|0:01:55|

