import matplotlib.pyplot as plt
import pandas as pd

playlists = pd.read_csv('donnees/playlists.data', sep='\t')
tracks = pd.read_csv('donnees/tracks.data', sep='\t')

# 2.1
# création dictionnaire
track_dict = {}
for i in range(0, tracks.index[-1]+1):
    val = []
    track_dict[tracks.at[i,'url']] = val

for i in range(0, playlists.index[-1]+1):
    track_dict[playlists.at[i,'url']].append(playlists.at[i, 'position'])

# création colonnes vides
tracks['pos_pic']=''
tracks['pic15']=''
tracks['livespan']=''
tracks['pos_avg']=''
tracks['avg15']=''

# remplissage des colonnes
for i in range(0, tracks.index[-1]+1):

    pic = min(track_dict[tracks.at[i, 'url']])
    tracks.at[i, 'pos_pic'] = pic

    tracks.at[i, 'pic15'] = int(pic < 15)
    
    livespan = len(track_dict[tracks.at[i, 'url']])
    tracks.at[i, 'livespan'] = livespan

    position_moyenne = sum(track_dict[tracks.at[i, 'url']])/livespan
    tracks.at[i, 'pos_avg'] = position_moyenne

    tracks.at[i, 'avg15'] = int(position_moyenne < 15)


# 2.2
subset = set([])
metal = playlists[playlists['playlist'] == 'metal']
for i in range(metal.index[0], metal.index[-1]+1):
    subset.add(metal.at[i, 'url'])

# df_subset = pd.DataFrame(subset)

'''TODO: - créer un dataframe contenant une fois chaque morceaux metal ->v df_metal,
         - utiliser le méthode get_dummies pour remplacer la variable catégorique key: pd.get_dummies(df_metal, key)
         - stocker le resultat dans un nouveau dataframe'''

metaltracks = tracks
for i in range(0, tracks.index[-1]+1):
    if(tracks.at[i, 'url'] not in subset):
        metaltracks = metaltracks.drop(i)

boxplot = metaltracks.boxplot()
plt.show()