import pandas as pd

playlists = pd.read_csv('donnees/playlists.data', sep='\t')
tracks = pd.read_csv('donnees/tracks.data', sep='\t')

track_dict = {}

for i in range(0, tracks.index[-1]+1):
    val = []
    track_dict[tracks.at[i,'url']] = val

for i in range(0, playlists.index[-1]+1):
    track_dict[playlists.at[i,'url']].append(playlists.at[i, 'position'])

for i in track_dict:
    print(track_dict[i])

