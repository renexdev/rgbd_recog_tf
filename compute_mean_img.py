import numpy as np
import cv2
import os, ipdb
import configure as cfg


def main():
    trial = 1
    train_dir = cfg.DIR_DATA
    train_lst = open(cfg.PTH_TRAIN_SHORT_LST[trial-1], 'r').read().splitlines()
    N = len(train_lst)

    mean_rgb = np.zeros((256,256,3))
    mean_dep = np.zeros((256,256,3))
    lim = 10
    for i in range(N):
        item = train_lst[i]

        im = cv2.imread(os.path.join(train_dir, item)+cfg.EXT_RGB, -1)
        mean_rgb += im

        im = cv2.imread(os.path.join(train_dir, item)+cfg.EXT_D, -1)
        mean_dep += im

        percent = int(100.0*i/N)
        if percent == lim:
            print '    Loaded %d / %d' % (i, N)
            lim += 10

    mean_rgb /= N
    mean_dep /= N
    np.save(cfg.PTH_RGB_MEAN, mean_rgb)
    np.save(cfg.PTH_DEP_MEAN, mean_dep)

    return

if __name__ == '__main__':
    main()
