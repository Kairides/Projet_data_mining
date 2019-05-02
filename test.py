import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import sklearn.preprocessing as sk
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

tracks = pd.get_dummies(tracks, columns=['Key'])
tracks = pd.get_dummies(tracks, columns=['Mode'], drop_first=True)

#Metal tracks
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
print(metaltracks.mean(axis=0))
print(sk.normalize(metaltracks.iloc[0:,2:]))
print('#######################\n')


#fr tracks
frset = set([])
fr = playlists[playlists['playlist'] == 'fr']
for i in range(fr.index[0], fr.index[-1]+1):
    frset.add(fr.at[i, 'url'])

frtracks = tracks

for i in range(0, tracks.index[-1]+1):
    if tracks.at[i, 'url'] not in frset:
        frtracks = frtracks.drop(i)

print('Chanson fr moyenne')
print(frtracks.mean(axis=0))
print('#######################\n')


#jazz tracks
jazzset = set([])
jazz = playlists[playlists['playlist'] == 'jazz']
jazz = jazz.reset_index(drop=True)#reindexation sinon problème car playlist coupée
for i in range(jazz.index[0], jazz.index[-1]+1):
    jazzset.add(jazz.at[i, 'url'])

jazztracks = tracks

for i in range(0, tracks.index[-1]+1):
    if tracks.at[i, 'url'] not in jazzset:
        jazztracks = jazztracks.drop(i)

print('Chanson jazz moyenne')
print(jazztracks.mean(axis=0))
print('#######################\n')


#lovepop tracks
popset = set([])
pop = playlists[playlists['playlist'] == 'lovepop']
for i in range(pop.index[0], pop.index[-1]+1):
    popset.add(pop.at[i, 'url'])

poptracks = tracks

for i in range(0, tracks.index[-1]+1):
    if tracks.at[i, 'url'] not in popset:
        poptracks = poptracks.drop(i)

print('Chanson pop moyenne')
print(poptracks.mean(axis=0))
print('#######################\n')


#electro tracks
electroset = set([])
electro = playlists[playlists['playlist'] == 'electro']
for i in range(electro.index[0], electro.index[-1]+1):
    electroset.add(electro.at[i, 'url'])

electrotracks = tracks

for i in range(0, tracks.index[-1]+1):
    if tracks.at[i, 'url'] not in electroset:
        electrotracks = electrotracks.drop(i)

print('Chanson electro moyenne')
print(electrotracks.mean(axis=0))
print('#######################\n')