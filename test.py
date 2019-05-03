import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import sklearn.preprocessing as sk
import scipy.spatial.distance as scp
import numpy as np

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
'''hmap = sb.heatmap(tracks.corr(), annot=True)
plt.show()'''

# affiche le morceau moyen d'un playlist


def closer_mean(norm_playlist):
    min_dist = [1000000, '']
    zero = np.full(25, 0)

    for i in range(1, len(norm_playlist)):
        dist = scp.euclidean(zero, norm_playlist[i])
        if dist < min_dist[0]:
            min_dist[0] = dist
            min_dist[1] = i

    return min_dist[1]

def print_mean_track(playlist):

    mean_track = playlist.mean(axis=0)

    for index in range(7):
        print(mean_track.index[index], ':', mean_track[index])

    key_max = ['', 0]
    for index in range(8, 19):
        if mean_track[index] > key_max[1]:
            key_max[0] = mean_track.index[index]
            key_max[1] = mean_track[index]

    print(key_max[0], key_max[1])

    if mean_track[20] > 0.5:
        print('Mode Mineur', mean_track[20])
    else:
        print('Mode Majeur', mean_track[20])


tracks = pd.get_dummies(tracks, columns=['Key'])
tracks = pd.get_dummies(tracks, columns=['Mode'], drop_first=True)

# Metal tracks
metalset = set([])
metal = playlists[playlists['playlist'] == 'metal']
for i in range(metal.index[0], metal.index[-1]+1):
    metalset.add(metal.at[i, 'url'])

metaltracks = tracks

for i in range(0, tracks.index[-1]+1):
    if tracks.at[i, 'url'] not in metalset:
        metaltracks = metaltracks.drop(i)

'''boxplot = metaltracks.boxplot()
plt.show()
plt.hist(metaltracks['BPM'], bins=50)
plt.show()
plt.hist(metaltracks['Danceability'], bins=90)
plt.show()
metaltracks.plot(x='Danceability', y=['Valence', 'BPM', 'Energy', 'Acousticness', 'Instrumentalness'], kind='bar')
plt.show()'''

print('Chanson metal moyenne')
print_mean_track(metaltracks)

norm_metal = (sk.normalize(metaltracks.iloc[0:, 2:]))
print(norm_metal)

morceau_metal_plus_moyen = closer_mean(norm_metal)

print(morceau_metal_plus_moyen)

print('#######################\n')


# fr tracks
frset = set([])
fr = playlists[playlists['playlist'] == 'fr']
for i in range(fr.index[0], fr.index[-1]+1):
    frset.add(fr.at[i, 'url'])

frtracks = tracks

for i in range(0, tracks.index[-1]+1):
    if tracks.at[i, 'url'] not in frset:
        frtracks = frtracks.drop(i)

print('Chanson fr moyenne')
print_mean_track(frtracks)

norm_fr = (sk.normalize(frtracks.iloc[0:, 2:]))

morceau_fr_plus_moyen = closer_mean(norm_fr)

print(morceau_fr_plus_moyen)
print('#######################\n')


# jazz tracks
jazzset = set([])
jazz = playlists[playlists['playlist'] == 'jazz']
jazz = jazz.reset_index(drop=True)  # reindexation sinon problème car playlist coupée
for i in range(jazz.index[0], jazz.index[-1]+1):
    jazzset.add(jazz.at[i, 'url'])

jazztracks = tracks

for i in range(0, tracks.index[-1]+1):
    if tracks.at[i, 'url'] not in jazzset:
        jazztracks = jazztracks.drop(i)

print('Chanson jazz moyenne')
print_mean_track(jazztracks)

norm_jazz = (sk.normalize(jazztracks.iloc[0:, 2:]))

morceau_jazz_plus_moyen = closer_mean(norm_jazz)

print(morceau_jazz_plus_moyen)
print('#######################\n')


# lovepop tracks
popset = set([])
pop = playlists[playlists['playlist'] == 'lovepop']
for i in range(pop.index[0], pop.index[-1]+1):
    popset.add(pop.at[i, 'url'])

poptracks = tracks

for i in range(0, tracks.index[-1]+1):
    if tracks.at[i, 'url'] not in popset:
        poptracks = poptracks.drop(i)

print('Chanson pop moyenne')
print_mean_track(poptracks)

norm_pop = (sk.normalize(poptracks.iloc[0:, 2:]))

morceau_pop_plus_moyen = closer_mean(norm_pop)
print(morceau_pop_plus_moyen)
print('#######################\n')


# electro tracks
electroset = set([])
electro = playlists[playlists['playlist'] == 'electro']
for i in range(electro.index[0], electro.index[-1]+1):
    electroset.add(electro.at[i, 'url'])

electrotracks = tracks

for i in range(0, tracks.index[-1]+1):
    if tracks.at[i, 'url'] not in electroset:
        electrotracks = electrotracks.drop(i)

print('Chanson electro moyenne')
print_mean_track(electrotracks)

norm_electro = (sk.normalize(electrotracks.iloc[0:, 2:]))

morceau_electro_plus_moyen = closer_mean(norm_electro)

print(morceau_electro_plus_moyen)
print('#######################\n')

