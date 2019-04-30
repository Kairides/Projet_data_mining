import pandas as pd

playlists = pd.read_csv('donnees/playlists.data', sep='\t')
tracks = pd.read_csv('donnees/tracks.data', sep='\t')

track_dict = {}

for i in range(0, tracks.index[-1]+1):
    val = []
    track_dict[tracks.at[i,'url']] = val

for i in range(0, playlists.index[-1]+1):
    track_dict[playlists.at[i,'url']].append(playlists.at[i, 'position'])

tracks['pos_pic']=''
tracks['pic15']=''
tracks['livespan']=''
tracks['pos_avg']=''
tracks['avg15']=''

for i in track_dict:
    print(track_dict[i])
    position_pic = min(track_dict[i])

    duree = len(track_dict[i])

    position_moyenne = sum(track_dict[i])/duree

    if position_pic < 15 :
        indic_pic = 1
    else:
        indic_pic = 0

    if position_moyenne < 15:
        indic_moyenne = 1
    else:
        indic_moyenne = 0

