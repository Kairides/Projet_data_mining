import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import sklearn.preprocessing as sk
import sklearn.decomposition as skdecomp
from sklearn import tree
import numpy as np
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error

playlists = pd.read_csv('donnees/playlists.data', sep='\t')
tracks = pd.read_csv('donnees/tracks.data', sep='\t')

# 2.1
# création dictionnaire
track_dict = {}
for i in range(0, tracks.index[-1]+1):
    val = []
    track_dict[tracks.at[i, 'url']] = val

for i in range(0, playlists.index[-1]+1):
    track_dict[playlists.at[i, 'url']].append(playlists.at[i, 'position'])

# création colonnes vides
tracks['pos_pic'] = ''
tracks['pic15'] = ''
tracks['livespan'] = ''
tracks['pos_avg'] = ''
tracks['avg15'] = ''

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

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Correlation entre les différents attributs', fontsize=20)
ax.set_xlabel('Attributs', fontsize=15)

hmap = sb.heatmap(tracks.corr(), annot=True)
plt.show()

# ################################# CUSTOM FUNCTIONS #############################################


# affiche le morceau moyen d'un playlist


def closer_mean(norm_playlist):
    min_dist = [1000000, '']
    norm_playlist = norm_playlist.values

    for j in range(0, len(norm_playlist)):
        dist = sum(np.power(norm_playlist[j], 2))**(1/2)
        if dist <= min_dist[0]:
            min_dist[0] = dist
            min_dist[1] = j

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
    for j in range(len(playlist)):
        this_moy = np.mean(track_dict[playlist.at[j, 'url']])
        if moy > this_moy:
            moy = this_moy
            best_track = playlist.at[j, 'url']

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

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Distributions des différents attributs pour la playlist Métal', fontsize=20)
ax.set_xlabel('Attributs', fontsize=15)

boxplot = metaltracks.iloc[:, 1:10].boxplot()
plt.show()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Distribution des morceaux pour l'attribut BPM dans la playlist Métal", fontsize=20)
ax.set_xlabel('Battement pour minute', fontsize=15)
ax.set_ylabel('Nombre de morceau', fontsize=15)

plt.hist(metaltracks['BPM'], bins=50)
plt.show()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Distribution des morceaux pour l'attribut Instrumentalness dans la playlist Métal", fontsize=20)
ax.set_xlabel('Instrumentalness')
ax.set_ylabel('Nombre de morceau')

plt.hist(metaltracks['Instrumentalness'], bins=90)
plt.show()

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

chop_suey_sorted.plot(x='date', y='position', kind='line')
plt.show()

# 2.3

acp = skdecomp.PCA(n_components=2)

features = ['BPM', 'Danceability', 'Valence', 'Energy', 'Acousticness', 'Instrumentalness', 'Liveness', 'Speechiness', 'Mode_Minor']

metaltracks['playlist'] = 'metal'
poptracks['playlist'] = 'lovepop'
electrotracks['playlist'] = 'electro'
jazztracks['playlist'] = 'jazz'
frtracks['playlist'] = 'fr'

tracks = pd.concat([metaltracks, poptracks, electrotracks, jazztracks, frtracks], ignore_index=True, sort=False)

x = tracks.loc[:, features].values
y = tracks.loc[:, ['playlist']] .values

x = sk.StandardScaler().fit_transform(x)

pd.DataFrame(data=x, columns=features)

composant = acp.fit_transform(x)
data_frame = pd.DataFrame(data=composant, columns=['1', '2'])

final = pd.concat([data_frame, tracks[['playlist']]], axis=1)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Premier composant principal')
ax.set_ylabel('Deuxieme composant principal')
ax.set_title('ACP sur toutes les playlists', fontsize=20)

playlists_list = ['metal', 'lovepop', 'jazz', 'electro', 'fr']
couleurs = ["red", "green", "blue", "magenta", "yellow"]

for playlist_list, couleur in zip(playlists_list, couleurs):
    indices = final['playlist'] == playlist_list
    ax.scatter(final.loc[indices, '1'], final.loc[indices, '2'], c=couleur, s=50)

ax.legend(playlists_list)
ax.grid()

plt.show()




#2.3 Modèles
print("###########################")
print("1er Modèle :")

X = tracks.loc[:, features]
y = pd.DataFrame(tracks['playlist'])
for i in range(0, y.index[-1]+1):
        playlist = y.at[i,'playlist']
        if(playlist == 'fr'):
                y.at[i,'playlist'] = 1
        elif playlist == 'jazz':
                y.at[i,'playlist'] = 2
        elif playlist == 'lovepop':
                y.at[i,'playlist'] = 3
        elif playlist == 'electro':
                y.at[i,'playlist'] = 4
        elif playlist == 'metal':
                y.at[i,'playlist'] = 5

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, train_size=0.2)

m = tree.DecisionTreeRegressor()
m.fit(X_train, y_train)
prediction = m.predict(X_test)
def evaluation(y_test, prediction):
    # The mean squared error
    print("Mean absolute error: %.2f"% mean_absolute_error(y_test, prediction))
    print("Median absolute error: %.2f"% median_absolute_error(y_test, prediction))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, prediction))

evaluation(y_test, prediction)
print("###########################\n")



print("###########################")
print("2eme Modèle :")

X = tracks.loc[:, features]
y = pd.DataFrame(tracks['pos_avg'])
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, train_size=0.2)

m = tree.DecisionTreeRegressor()
m.fit(X_train, y_train)
prediction = m.predict(X_test)
evaluation(y_test, prediction)
print(m.predict([metaltracks.loc[1, features]]))
print(metaltracks.at[1, 'pos_avg'])
print("###########################\n")
