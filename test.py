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

for i in range(0, tracks.index[-1]+1):

    min = min(track_dict[tracks.at[i, 'url']])
    tracks.at[i, 'pos_pic'] = min

    tracks.at[i, 'pic15'] = int(min < 15)
    
    livespan = len(track_dict[tracks.at[i, 'url']])
    tracks.at[i, 'livespan'] = livespan

    position_moyenne = sum(track_dict[tracks.at[i, 'url']])/livespan
    tracks.at[i, 'pos_avg'] = position_moyenne

    tracks.at[i, 'avg15'] = int(position_moyenne < 15)

print(tracks)