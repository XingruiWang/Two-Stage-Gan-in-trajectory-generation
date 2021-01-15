import webbrowser
from transform import transfer
def draw_traj(trajs, output_file_name = 'map.html', if_append = False):
    mode = 'a' if if_append else 'w'
    with open('template/template.html', 'r') as f1, open(output_file_name, mode) as f2:
        read_lines = f1.readlines() # 整行读取数据
        write = ''
        for line in read_lines:
            l = line.strip()
            if l == '//add here':
                write += '//add here\n'
                for traj in trajs:
                    write += '\tvar pois = [\n'
                    for pos in traj:
                        write = write + '\t\tnew BMap.Point('+str(pos[1])+','+str(pos[0])+'),\n'
                    write = write[:-2]+'\n\t];\n\ttraj.push(pois);\n'
            else:
                write += line

        f2.write(write)
        f1.close()
        f2.close()
    webbrowser.open(output_file_name)

if __name__ == '__main__':
    import numpy as np
    tr = transfer()
    d = np.load('D:/文档/6_项目/3_荣誉辅修/数据/Data/000/Trajectory/20090705025307.npy', allow_pickle=True)
    e = np.load('D:/文档/6_项目/3_荣誉辅修/数据/Data/001/Trajectory/20081214080334.npy', allow_pickle=True)
    for i in range(len(d)):
        d[i] = [tr.wg84_to_bd09(float(d[i,1]),float(d[i,0]))[1],tr.wg84_to_bd09(float(d[i,1]),float(d[i,0]))[0]]
    for i in range(len(e)):
        e[i] = [tr.wg84_to_bd09(float(e[i,0]),float(e[i,0]))[1],tr.wg84_to_bd09(float(e[i,1]),float(e[i,0]))[0]]

    draw_traj([d,e])


