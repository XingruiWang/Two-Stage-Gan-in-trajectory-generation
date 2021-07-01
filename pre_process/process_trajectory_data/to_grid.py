import numpy as np


def to_grid(traj, min_lon, min_lat, d=0.01, is_num=False):
    """
    traj: 单条轨迹数据，ny.array格式
    min_lon, min_lat: 经度、纬度起始值
    is_num： d是否指格点数
    d: 格点数或格点边长
    """
    if is_num:
        grids_num = d
        d = max(max(traj[:, 0])-min_lon, max(traj[:, 1])-min_lat)/grids_num
        print(d)
    else:
        grids_num = int(max(max(traj[:, 0])-min_lon, max(traj[:, 1])-min_lat)//d) + 1

    traj_mat_1 = np.zeros(shape=(grids_num, grids_num))
    traj_mat_2 = np.zeros(shape=(grids_num, grids_num))

    t = np.array(range(0, traj.shape[0]*15, 15))
    t1 = np.column_stack((traj, t))
    t2 = []
    t3 = []

    for i in t1:
        a = int((float(i[1])-lat)//d)
        b = int((float(i[0])-lon)//d)
        t2.append([a, b, i[2]])

    for i in t2:
        if len(t3) == 0:
            t3.append(i)
        else:
            if t3[-1][0:2] != i[0:2]:
                t3.append(i)

    for i in range(len(t3)-1):
        t3[i][2] = t3[i+1][2]-t3[i][2]

    t3[-1][2] = t1[-1][2]-t3[-1][2]

    for point in t3:
        print(point)
        if traj_mat_1[-point[0], point[1]] == 0:
            traj_mat_1[-point[0], point[1]] = point[2]
        elif traj_mat_2[-point[0], point[1]] == 0:
            traj_mat_2[-point[0], point[1]] = point[2]
        else:
            raise ValueError(
                'Data error. Pass through a point more than twice.')

    return (traj_mat_1, traj_mat_2)


```
# example 
tj = np.load('./data/9.npy', allow_pickle=True)
m1, m2 = to_grid(tj, -8.62, 41.13, 10, True)
# m1: the first channel of the trajectory map
# m2: the second channel of the trajectory map
# the samilar processing methods as Ouyang et al. (2018)
# (Ouyang K, Shokri R, Rosenblum DS, Yang W (2018). A non-parametric generative model for human trajectories. In: International Joint Conferences on Artificial Intelligence, (J Lang, eds.), 3812–3817)
``` 
