***Google Colab -> Cette cellule est à executer (1 fois) pour le cloner le dépot en local***


```python
!git clone https://github.com/ManDes71/Rakuten_Text_Classification_ML.git
```

    Cloning into 'Rakuten_Text_Classification_ML'...
    remote: Enumerating objects: 315, done.[K
    remote: Counting objects: 100% (107/107), done.[K
    remote: Compressing objects: 100% (101/101), done.[K
    remote: Total 315 (delta 55), reused 16 (delta 5), pack-reused 208[K
    Receiving objects: 100% (315/315), 52.00 MiB | 7.32 MiB/s, done.
    Resolving deltas: 100% (146/146), done.
    Updating files: 100% (56/56), done.
    Filtering content: 100% (5/5), 1.41 GiB | 43.60 MiB/s, done.
    

***Google Colab -> Cette cellule est à executer (2 fois) pour installer les bibliothèques nécessaires***  
You must restart the runtime in order to use newly installed versions.  


```python
import sys
sys.path.append('/content/Rakuten_Text_Classification_ML')
import sys
print(sys.version)

!pip install -r /content/Rakuten_Text_Classification_ML/requirements.txt
```

    3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0]
    Requirement already satisfied: numpy==1.26.3 in /usr/local/lib/python3.10/dist-packages (from -r /content/Rakuten_Text_Classification_ML/requirements.txt (line 1)) (1.26.3)
    Requirement already satisfied: pandas==1.5.3 in /usr/local/lib/python3.10/dist-packages (from -r /content/Rakuten_Text_Classification_ML/requirements.txt (line 2)) (1.5.3)
    Requirement already satisfied: matplotlib==3.8.0 in /usr/local/lib/python3.10/dist-packages (from -r /content/Rakuten_Text_Classification_ML/requirements.txt (line 3)) (3.8.0)
    Requirement already satisfied: scikit_learn==1.2.2 in /usr/local/lib/python3.10/dist-packages (from -r /content/Rakuten_Text_Classification_ML/requirements.txt (line 4)) (1.2.2)
    Requirement already satisfied: joblib==1.2.0 in /usr/local/lib/python3.10/dist-packages (from -r /content/Rakuten_Text_Classification_ML/requirements.txt (line 5)) (1.2.0)
    Requirement already satisfied: langdetect==1.0.9 in /usr/local/lib/python3.10/dist-packages (from -r /content/Rakuten_Text_Classification_ML/requirements.txt (line 6)) (1.0.9)
    Requirement already satisfied: nltk==3.7 in /usr/local/lib/python3.10/dist-packages (from -r /content/Rakuten_Text_Classification_ML/requirements.txt (line 7)) (3.7)
    Requirement already satisfied: xgboost==1.7.3 in /usr/local/lib/python3.10/dist-packages (from -r /content/Rakuten_Text_Classification_ML/requirements.txt (line 8)) (1.7.3)
    Requirement already satisfied: wordcloud==1.9.2 in /usr/local/lib/python3.10/dist-packages (from -r /content/Rakuten_Text_Classification_ML/requirements.txt (line 9)) (1.9.2)
    Requirement already satisfied: seaborn==0.12.2 in /usr/local/lib/python3.10/dist-packages (from -r /content/Rakuten_Text_Classification_ML/requirements.txt (line 10)) (0.12.2)
    Requirement already satisfied: scipy==1.11 in /usr/local/lib/python3.10/dist-packages (from -r /content/Rakuten_Text_Classification_ML/requirements.txt (line 11)) (1.11.0)
    Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas==1.5.3->-r /content/Rakuten_Text_Classification_ML/requirements.txt (line 2)) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas==1.5.3->-r /content/Rakuten_Text_Classification_ML/requirements.txt (line 2)) (2023.3.post1)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib==3.8.0->-r /content/Rakuten_Text_Classification_ML/requirements.txt (line 3)) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib==3.8.0->-r /content/Rakuten_Text_Classification_ML/requirements.txt (line 3)) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib==3.8.0->-r /content/Rakuten_Text_Classification_ML/requirements.txt (line 3)) (4.47.2)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib==3.8.0->-r /content/Rakuten_Text_Classification_ML/requirements.txt (line 3)) (1.4.5)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib==3.8.0->-r /content/Rakuten_Text_Classification_ML/requirements.txt (line 3)) (23.2)
    Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib==3.8.0->-r /content/Rakuten_Text_Classification_ML/requirements.txt (line 3)) (9.4.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib==3.8.0->-r /content/Rakuten_Text_Classification_ML/requirements.txt (line 3)) (3.1.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit_learn==1.2.2->-r /content/Rakuten_Text_Classification_ML/requirements.txt (line 4)) (3.2.0)
    Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from langdetect==1.0.9->-r /content/Rakuten_Text_Classification_ML/requirements.txt (line 6)) (1.16.0)
    Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from nltk==3.7->-r /content/Rakuten_Text_Classification_ML/requirements.txt (line 7)) (8.1.7)
    Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.10/dist-packages (from nltk==3.7->-r /content/Rakuten_Text_Classification_ML/requirements.txt (line 7)) (2023.6.3)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from nltk==3.7->-r /content/Rakuten_Text_Classification_ML/requirements.txt (line 7)) (4.66.1)
    

# ***PROJET RAKUTEN***  
# **1) Description du projet**  
**Description du problème**  
L'objectif de ce défi est la classification à grande échelle des données de produits multimodales (texte et image) en type de produit.  
Par exemple, dans le catalogue de Rakuten France, un produit avec une désignation "Grand Stylet Ergonomique Bleu Gamepad Nintendo Wii U - Speedlink Pilot Style" associé à une image (image_938777978_product_201115110.jpg) et
parfois à une description supplémentaire. Ce produit est catégorisé sous le code de produit 50.


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


```python
df.head()
```





  <div id="df-7b857ba5-d143-4c99-9fc0-21940cd22a1c" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7b857ba5-d143-4c99-9fc0-21940cd22a1c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7b857ba5-d143-4c99-9fc0-21940cd22a1c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7b857ba5-d143-4c99-9fc0-21940cd22a1c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-9f67380d-13cf-4564-a97d-47957606d340">
  <button class="colab-df-quickchart" onclick="quickchart('df-9f67380d-13cf-4564-a97d-47957606d340')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-9f67380d-13cf-4564-a97d-47957606d340 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





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
    


    
![png](ReadMe_ML_files/ReadMe_ML_8_1.png)
    


Ce notebook fait partie d'un ensemble de sous-projets dont le resultat représente le **projet Rakuten** que j'ai réalisé pour mon diplôme de data Scientist chez Datascientest.com.  
Ce projet consiste en la classification à grande échelle des données de         produits multimodales (texte et image) en type de produits.  
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
# **2) Introduction**  
le but du projet est de prédire le code de type de chaque produit tel que défini dans le catalogue de Rakuten France.  
La catégorisation des annonces de produits se fait par le biais de la désignation, de la description (quand elle est présente) et des images.  
Les fichiers de données sont distribués ainsi :  
***X_train_update.csv*** : fichier d'entrée d'entraînement  
***Y_train_CVw08PX.csv*** : fichier de sortie d'entraînement  
***X_test_update.csv*** : fichier d'entrée de test  
Un fichier images.zip est également fourni, contenant toutes les images.  
La décompression de ce fichier fournira un dossier nommé "images" avec deux sous-dossiers nommés ***"image_train"*** et ***"image_test"***, contenant respectivement les images d'entraînement et de test.  
Pour notre part, ne participant pas au challenge Rakuten, je n'ai pas pas accès au fichier de sortie de test.  
Le fichier d’entrée de test est donc inutilisable.  
**X_train_update.csv** : fichier d'entrée d'entraînement :  
La première ligne des fichiers d'entrée contient l'en-tête et les colonnes sont séparées par des virgules (",").  
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

La liaison entre les fichiers se fait par une jointure sur l’identifiant entier présent les deux
fichiers.

# ***exploration du dataset.***  
## Examinons la répartition  des codes produits :


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


    
![png](ReadMe_ML_files/ReadMe_ML_12_0.png)
    


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




```python
print("----df_feats info-------")
print(df_feats.info())
print("-"*50)
print("Le champ description n'est pas toujours présent.")
print("-"*50)
# Calcul de la moyenne des longueurs pour chaque colonne séparément
moyenne_designation = df_feats['designation'].str.len().mean()
moyenne_description = df_feats['description'].str.len().mean()

print("Moyenne de la longueur des designations:", moyenne_designation)
print("Moyenne de la longueur des descriptions:", moyenne_description)

```

    ----df_feats info-------
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 84916 entries, 0 to 84915
    Data columns (total 5 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   Unnamed: 0   84916 non-null  int64 
     1   designation  84916 non-null  object
     2   description  55116 non-null  object
     3   productid    84916 non-null  int64 
     4   imageid      84916 non-null  int64 
    dtypes: int64(3), object(2)
    memory usage: 3.2+ MB
    None
    --------------------------------------------------
    Le champ description n'est pas toujours présent.
    --------------------------------------------------
    Moyenne de la longueur des designations: 70.16330255782185
    Moyenne de la longueur des descriptions: 808.1716924305102
    


```python
import matplotlib.pyplot as plt

categories = ['Designation', 'Description']

moyennes = [moyenne_designation, moyenne_description]

plt.figure(figsize=(16, 6))

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


    
![png](ReadMe_ML_files/ReadMe_ML_16_0.png)
    


## Examinons les valeurs nulles et les doublons du champ 'designation':


```python
categories = ['Non nulles', 'Uniques']
nb_designation = len(df_feats[~df_feats['designation'].isna()])
nb_designation_u = len(df_feats['designation'].unique())

Nb = [nb_designation, nb_designation_u]

plt.figure(figsize=(8, 6))  # Vous pouvez ajuster la taille selon vos besoins
plt.bar(categories,Nb, color=['blue', 'green'])  # Choisir des couleurs

plt.title('valeurs non nulles et unicité du champ  Designation')
plt.xlabel('Désignation')
plt.ylabel('Nombres de produits')
plt.xticks(categories)

plt.show()

```


    
![png](ReadMe_ML_files/ReadMe_ML_18_0.png)
    



```python

```


```python

```

## Examinons les valeurs nulles et les doublons du champ 'description'.


```python
categories = ['Non nulles', 'Uniques']
nb_description = len(df_feats[~df_feats['description'].isna()])
nb_description_u = len(df_feats['description'].unique())

Nb = [nb_description, nb_description_u]

plt.figure(figsize=(8, 6))
plt.bar(categories,Nb, color=['blue', 'green'])

plt.title('valeurs non nulles et unicité du champ  Description')
plt.xlabel('Description')
plt.ylabel('Nombres de produits')
plt.xticks(categories)

plt.show()
```


    
![png](ReadMe_ML_files/ReadMe_ML_22_0.png)
    

