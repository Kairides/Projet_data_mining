import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import sklearn.preprocessing as sk
import sklearn.decomposition as skdecomp
import numpy as np

playlists = pd.read_csv('donnees/playlists.data', sep='\t')
tracks = pd.read_csv('donnees/tracks.data', sep='\t')

# 2.1
# création dictionnaire
track_dict = {}
for i in range(0, tracks.index[-1]+1):
    val = []
    track_dict[tracks.at[i, 'url']] = val

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

# ################################# CUSTOM FUNCTIONS #############################################


# affiche le morceau moyen d'un playlist


def closer_mean(norm_playlist):
    min_dist = [1000000, '']
    zero = np.full(len(norm_playlist), 0)
    norm_playlist = norm_playlist.values

    for i in range(0, len(norm_playlist)):
        dist = sum(np.power(norm_playlist[i], 2))**(1/2)
        if dist <= min_dist[0]:
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


def print_closest_mean(playlist):
    playlist_copy = playlist.drop(['url', 'pos_pic', 'pic15', 'livespan', 'pos_avg', 'avg15'], axis=1)
    norm_playlist = (playlist_copy.sub(playlist_copy.mean(axis=0))).div(playlist_copy.std(axis=0))

    morceau_plus_moyen = closer_mean(norm_playlist)
    # playlist = playlist.reset_index(drop=True)

    print("\nmorceau plus moyen: ", playlist.at[morceau_plus_moyen, 'url'])


def print_best_track(playlist):
    moy = 100
    best_track = ''
    for i in range(len(playlist)):
        this_moy = np.mean(track_dict[playlist.at[i, 'url']])
        if moy > this_moy:
            moy = this_moy
            best_track = playlist.at[i, 'url']

    print("Morceau mieux classé", best_track)


#######################################################################################################

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

print('Chanson metal moyenne :')
print_mean_track(metaltracks)

metaltracks = metaltracks.reset_index(drop=True)

print_closest_mean(metaltracks)

print_best_track(metaltracks)

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

print('Chanson fr moyenne :')
print_mean_track(frtracks)

frtracks = frtracks.reset_index(drop=True)

print_closest_mean(frtracks)

print_best_track(frtracks)

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

print('Chanson jazz moyenne:')
print_mean_track(jazztracks)

jazztracks = jazztracks.reset_index(drop=True)

print_closest_mean(jazztracks)

print_best_track(jazztracks)

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

print('Chanson pop moyenne:')
print_mean_track(poptracks)

poptracks = poptracks.reset_index(drop=True)

print_closest_mean(poptracks)

print_best_track(poptracks)

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

print('Chanson electro moyenne:')
print_mean_track(electrotracks)

electrotracks = electrotracks.reset_index(drop=True)

print_closest_mean(electrotracks)

print_best_track(electrotracks)

print('#######################\n')

# On cherche l'évolution du classemement de Chop Suey

chop_suey_chart = metal[metal['url'] == metaltracks.at[234, 'url']]
chop_suey_sorted = chop_suey_chart.sort_values('date')

''' chop_suey_sorted.plot(x='date', y='position', kind='line')
plt.show() '''


# 2.3

acp = skdecomp.PCA(n_components=2)

# Essais d'ACP sur les morceaux metal
'''metal_acp = acp.fit_transform(metaltracks.iloc[:, 2:11])

plt.scatter(metal_acp[:, 0], metal_acp[:, 1])

plt.show()'''


# Essais d'ACP sur tous les morceaux
total_acp = acp.fit_transform(tracks.iloc[:, 2:11])

plt.scatter(total_acp[:, 0], total_acp[:, 1])

plt.show()
