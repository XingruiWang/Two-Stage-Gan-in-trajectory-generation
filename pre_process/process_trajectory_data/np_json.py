import numpy as np
import json

def np2json(traj):
    out = []
    start_time = 1573617600
    for i in traj:
        d = {"longitude": float(i[1]), "latitude": float(i[0]), "coord_type_input": "wgs84", "loc_time": start_time}
        start_time +=  5
        out.append(d)
    out = json.dumps( out )

    return out

def json2np(traj_json):
    print(json.load(traj_json[0]))


