import os, glob, ipdb
import configure as cfg
import numpy as np
import cv2

IMREAD_COLOR = int(cv2.IMREAD_COLOR)
IMREAD_UNCHANGED = int(cv2.IMREAD_UNCHANGED)


IMSIZE = (256, 256)
def scaleit3(img):
    imsz = img.shape

    # rescale
    rd = float(IMSIZE[0]) / float(np.max(imsz))
    imrange_rescale = cv2.resize(img, (int(rd*imsz[1]), int(rd*imsz[0])), interpolation=cv2.INTER_CUBIC)
    imszn = imrange_rescale.shape

    nchan = 1
    if len(imszn)==3:
        nchan = imszn[2]

    # fill it
    imgcanvas = np.zeros( (IMSIZE[0],IMSIZE[1],nchan), dtype='uint8' )
    offs_col = (IMSIZE[1] - imszn[1]) / 2
    offs_row = (IMSIZE[0] - imszn[0]) / 2

    # take cols
    imgcanvas[offs_row:offs_row+imszn[0], offs_col:offs_col+imszn[1]] = imrange_rescale.reshape( (imszn[0],imszn[1],nchan) )
    return (imgcanvas)


def scaleit_experimental(img):
    imsz = img.shape
    mxdim  = np.max(imsz)
    
    offs_col = (mxdim - imsz[1])/2
    offs_row = (mxdim - imsz[0])/2
    nchan = 1
    if(len(imsz)==3):
        nchan = imsz[2]
    imgcanvas = np.zeros(  (mxdim,mxdim,nchan), dtype='uint8' )
    imgcanvas[offs_row:offs_row+imsz[0], offs_col:offs_col+imsz[1]] = img.reshape( (imsz[0],imsz[1],nchan) )

    # take rows
    if(offs_row):
        tr = img[0,:]
        br = img[-1,:]
        imgcanvas[0:offs_row,:,0] = np.tile(tr, (offs_row,1))
        imgcanvas[-offs_row-1:,:,0] = np.tile(br, (offs_row+1,1)) 

    # take cols
    if(offs_col):
        lc = img[:,0]
        rc = img[:,-1]
        imgcanvas[:, 0:offs_col,0] = np.tile(lc, (offs_col,1)).transpose()
        imgcanvas[:, -offs_col-1:,0] = np.tile(rc, (offs_col+1,1)).transpose()

    # rescale
    imrange_rescale = cv2.resize(imgcanvas, IMSIZE, interpolation=cv2.INTER_CUBIC)
    return imrange_rescale


def process_multi(dir_input, dir_output, masked):
    # load mean image
    mean_img = np.load(cfg.PTH_MEAN_IMG)
    mean_img = mean_img.transpose(1,2,0)

    # load all categories
    with open(cfg.PTH_DICT, 'r') as f: categories = f.read().splitlines()
    
    if not os.path.exists(dir_output): os.mkdir(dir_output)
    # go through the whole dataset
    for category in categories:
        print category
        dir_category = os.path.join(dir_input, category)
        dir_category_out = os.path.join(dir_output, category)
        if not os.path.exists(dir_category_out): os.mkdir(dir_category_out)

        # retrieve all objects in the category
        objects = os.listdir(dir_category)
        objects.sort()
        for obj in objects:
            if obj.startswith('.'): continue
            dir_obj = os.path.join(dir_category, obj)
            dir_obj_out = os.path.join(dir_category_out, obj)
            if not os.path.exists(dir_obj_out): os.mkdir(dir_obj_out)

            # get all instances of an object, without extension
            instances = glob.glob1(dir_obj, '*_loc.txt')
            instances = [instance.replace('_loc.txt','') for instance in instances]
            instances.sort()

            for instance in instances:
                # load images
                rgb = cv2.imread(os.path.join(dir_obj, instance+cfg.EXT_RGB), -1)
                dep = cv2.imread(os.path.join(dir_obj, instance+cfg.EXT_D), -1)

                if masked:
                    mask = cv2.imread(os.path.join(dir_obj, instance+cfg.EXT_MASK), -1)
                    if mask is None:
                        foo = glob.glob(os.path.join(dir_obj, instance+'_*'))
                        for bar in foo:
                            bar = bar[bar.rindex('/')+1:]
                            old = os.path.join(dir_obj, bar)
                            new = os.path.join(cfg.DIR_DATA_AUX, bar)
                            os.rename(old, new)
                        continue
                    mask /= 255
                    mask3 = np.dstack((mask,mask,mask))
                    rgb *= mask3
                    dep *= mask

                # scale
                #rgb = scaleit3(rgb)
                b = scaleit_experimental(rgb[:,:,0])
                g = scaleit_experimental(rgb[:,:,1])
                r = scaleit_experimental(rgb[:,:,2])
                rgb = np.dstack((b,g,r))


                dep = scaleit_experimental(dep)
                dep = cv2.applyColorMap(dep, cv2.COLORMAP_JET)

                # save images
                cv2.imwrite(os.path.join(dir_obj_out, instance+cfg.EXT_RGB), rgb)
                cv2.imwrite(os.path.join(dir_obj_out, instance+cfg.EXT_D), dep)
    return


if __name__ == '__main__':
    dir_input = cfg.DIR_DATA_RAW
    dir_output = cfg.DIR_DATA
    masked = False


    #dir_input = cfg.DIR_DATA_EVAL_RAW
    #dir_output = cfg.DIR_DATA_EVAL
    #masked = False


    #dir_input = cfg.DIR_DATA_RAW
    #dir_output = cfg.DIR_DATA_MASKED
    #masked = True

    print 'Input directory: %s' % dir_input
    print 'Output directory: %s' % dir_output
    process_multi(dir_input, dir_output, masked)
