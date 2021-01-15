from generate_traj_plot import draw_traj
import requests
import webbrowser
import numpy as np
from transform import transfer
from np_json import np2json, json2np


def baidu_smooth(trajs, key):
    r = requests.post('http://api.map.baidu.com/rectify/v1/track?',
                      data={'ak': key, 'point_list': trajs,'rectify_option':'need_mapmatch:1|transport_mode:auto|denoise_grade:5|vacuate_grade:1'})
    #points = req.json().get('points')
    if r.json().get('status') != 0:
        raise ValueError("Smoothing failed(%s). %s" %(r.json().get('status'),r.json().get('message')))
    return r.json().get('points')


class Traj():
    def __init__(self, ak=None, trajs=[], smooth=False):
        if ak is None:
            raise ValueError(
                "Your AK is require, since this package is base on Baidu Map API")
        self.ak = ak
        self.trajs = []
        if smooth:
            for i in trajs:
                self.trajs.append[self.__smooth(t)]
        else:
            self.trajs = trajs[:]

    def __iter__(self, i):
        return self.trajs[i]

    def append_traj(self, traj):
        self.trajs.append(traj)

    def __plot(self, traj, file, to_transform):
        if to_transform:
            traj = self.__transform(traj)
        draw_traj([traj],file)

    def __plot_list(self, traj, file, to_transform):
        if to_transform:
            traj = self.__transform_list(traj)
        draw_traj(traj, file)

    def plot(self, traj=None, file = 'map.html',to_transform=False):
        if traj is None:
            self.__plot_list(self.trajs, file, to_transform)
        elif type(traj).__name__ == 'list':
            print('1')
            self.__plot_list(traj, file, to_transform)
        else:
            self.__plot(traj, file, to_transform)

    def __smooth(self, traj):
        traj_json = np2json(traj)
        smooth_traj = baidu_smooth(traj_json, self.ak)
        out = []
        for p in smooth_traj:

            out.append([p['latitude'], p['longitude']])
        out = np.array(out)
        return out

    def __smooth_list(self, traj):
        out = []
        l = len(traj)
        i = 1
        for t in traj:
            print('%d / %d' %(i, l),end = '\n')
            out.append(self.__smooth(t))
            i += 1
        return out

    def smooth(self, traj=None):
        print('Smoothing ...')
        if traj is None:
            self.trajs = self.__smooth_list(self.trajs)
        elif type(traj).__name__ == 'list':
            return self.__smooth_list(traj)
        else:
            return self.__smooth(traj)

    def __transform(self, t):
        t = t[:]
        tr = transfer()
        l = len(traj)
        i = 1
        for i in range(len(t)):
            print('%d / %d' %(i, l),end = '\n')
            t[i] = [tr.wg84_to_bd09(float(t[i, 1]), float(t[i, 0]))[1], tr.wg84_to_bd09(float(t[i, 1]), float(t[i, 0]))[0]]
        return t

    def __transform_list(self, trajs):
        out = []
        for t in trajs:
            out.append(self.__transform(t))
        out = np.array(out)
        return out

    def transform(self, t=None):
        print('Transforming ...')
        if t is None:
            self.trajs = self.__transform_list(self.trajs)
        elif type(t).__name__ == 'list':
            return self.__transform_list(t)
        else:
            return self.__transform(t)


def main():
    t1 = np.load(
        '/path/to/traj/20090705025307.npy', allow_pickle=True)
    t2 = np.load(
        '/path/to/traj/20081023124523.npy', allow_pickle=True)

    print(t1)
    traj = Traj('DofzdMRY0OEZeKlQgocFFYXlhVNLAxpa', [t1, t2])

    traj.plot(file = 'map.html')


if __name__ == '__main__':
    main()
