import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import shutil
import os

def data_field(data, 
               name,
               show = False,
               header = 0) :
    ''' Take from an EBSD txt file a parameter and plot it.
    
    Args:
       data : str
           txt file
       name : str
           name of parameter to pick in DataFrame
       show : Boolean
           Show or not the output picture
       header : int
           Header of the txt file --> pd.read_csv
    Returns:
       data : np.ndarray
           Picture of parameter chose
    '''
    data = pd.read_csv(data, header = header, sep="\t|,")
    np_data = data.to_numpy()
    Columns = data.columns
    for i in range (len(Columns)) :
        if Columns[i] == 'x' :
            xi = i
        if Columns[i] == 'y' :
            yi = i
    dx = 0
    for i in range (len(Columns)) :
        if name == Columns[i] :
            dx = np_data[1,xi] - np_data[0,xi]
            if dx == 0 :
                dy = np_data[1,yi] - np_data[0,yi]
                dx = dy
                ly = (np.max(abs(np_data[:,xi]))/dx+1).astype(int)
                lx = (np.max(abs(np_data[:,yi]))/dy+1).astype(int)
                data = np.transpose(np_data[:,i].reshape((ly,lx)))
            else :
                dy = dx
                lx = (np.max(abs(np_data[:,xi]))/dx+1).astype(int)
                ly = (np.max(abs(np_data[:,yi]))/dy+1).astype(int)
                data = np_data[:,i].reshape((ly,lx))
        else :
            ()
    if dx == 0 :
        print('Not any constant named ', name, '. /n Try in this list : ', Columns)
    else :
        if show :
            plt.imshow(data)
            plt.colorbar()
            plt.title(name)
        return (data)

def clear_border(im) :
    nx, ny = im.shape
    for i in range(nx) :
        tr = np.where(im[i, :]!=0)[0]
        if len(tr) < 2 :
            ()
        else :
            im[i, tr[0]] = 0
            im[i, tr[-1]] = 0
        
    for j in range(ny) :
        tr = np.where(im[:, j]!=0)[0]
        if len(tr) < 2 :
            ()
        else :
            im[tr[0], j] = 0
            im[tr[-1], j] = 0        
    return(im)

def nd_addition (im1,
                 im2) :
    x,y = im1.shape
    im = im1 + im2
    for i in range (x) :
        for j in range (y) :
            if im1[i,j] != 0 and im2[i,j] != 0 :
                # im[i,j] = (im1[i,j] + im2[i,j])/2
                im[i,j] = im1[i,j]
                
    return(im)

def save_big_file_FCC (npfiles,
                       npnames,
                       model = 'Model.ctf',
                       namefile = '2_EBSD_BIG.ctf') :
    print('BIG FILE saveprocessing')

    shutil.copyfile(model, namefile)
    f = open(namefile, 'r')
    lines = f.readlines()
    last_line = lines[-1]
    last_elements = last_line.split('\t')
    last_elements[-1] = last_elements[-1][:-1]
    f.close()
    
    last_elements.remove('Phase')
    last_elements.remove('X')
    last_elements.remove('Y')
    
    N = len(npfiles)
    ny, nx = npfiles[0].shape
    npfiles_arrange = np.zeros((N, ny, nx))
    for i in range(N) :
        for j in range (N) :
            if npnames[j].lower() == last_elements[i].lower() :
                npfiles_arrange[i] = npfiles[j]
    
    with open(namefile, "a") as f :
        for i in range (nx) :
            for j in range (ny) :
                l = [2, i, j]
                for line in range (N) :
                    l.append(round(npfiles_arrange[line,j, i], 2))
                for word in l:
                      f.write(str(word) + '\t')
                f.write('\n')
            if i%100 == 0 :
                print(i, '/', nx)
    f.close()
    print('BIG FILE saved as ', namefile)

def load_correspondence_pts(pts_path):
    """
    Loads correspondence points between EBSD and microscope coordinate systems
    Points are to be generated from FIJI "Coordinate Picker Tool" as CSV files
    (https://imagej.nih.gov/ij/macros/tools/CoordinatePickerTool.txt, save it in macros/toolsets as txt file)
    CSV files must share the same name pathname structure
    
    Arguments
    ---------
    pts_path : str
        Pathname which precedes all file points filepaths
    
    Returns
    -------
    correspondence_pts : np.ndarray, 2D, shape=(pts_number, 2)
        (y,x) position of correspondence points in the file coordinate system
    """
    f = open(pts_path, 'r')
    Lines = f.readlines()
    correspondence_pts = np.zeros((len(Lines),2))
    for v, line in enumerate(Lines) :
        L = line.strip()
        e = int(0)
        for i in range(len(L)) :
            t = L[i]
            if t =='\t':
                correspondence_pts[v,1] = float(L[int(e):int(i)])
                e=i+1
            elif i+1 == len(L) :
                correspondence_pts[v,0] = float(L[int(e):int(i)])
    return correspondence_pts

def _make_inverse_warp(from_points, to_points, output_region, approximate_grid):
    x_min, y_min, x_max, y_max = output_region
    if approximate_grid is None : 
        approximate_grid = 1
    x_steps = (x_max - x_min) // approximate_grid
    y_steps = (y_max - y_min) // approximate_grid
    x, y = np.mgrid[x_min:x_max:x_steps*1j, y_min:y_max:y_steps*1j]

    # make the reverse transform warping from the to_points to the from_points, because we
    # do image interpolation in this reverse fashion
    def _U(x):
        return (x**2) * np.where(x<1e-100, 0, np.log(x))
    def _calculate_f(coeffs, points, x, y):
        w = coeffs[:-3]
        a1, ax, ay = coeffs[-3:]
        summation = np.zeros(x.shape)
        for wi, Pi in zip(w, points):
            summation += wi * _U(np.sqrt((x-Pi[0])**2 + (y-Pi[1])**2))
        return a1 + ax*x + ay*y + summation
    to_points, from_points = np.asarray(to_points), np.asarray(from_points)
    err = np.seterr(divide='ignore')
    
    # Make L matrix
    n = len(to_points)
    
    # Interpoints_distances
    xd = np.subtract.outer(to_points[:,0], to_points[:,0])
    yd = np.subtract.outer(to_points[:,1], to_points[:,1])
    K = _U(np.sqrt(xd**2 + yd**2))
    P = np.ones((n, 3))
    P[:,1:] = to_points
    O = np.zeros((3, 3))
    L = np.asarray(np.bmat([[K, P],[P.transpose(), O]]))

    V = np.resize(from_points, (len(from_points)+3, 2))
    V[-3:, :] = 0
    coeffs = np.dot(np.linalg.pinv(L), V)
    x_warp = _calculate_f(coeffs[:,0], to_points, x, y)
    y_warp = _calculate_f(coeffs[:,1], to_points, x, y)
    np.seterr(**err)
    transform = [x_warp, y_warp]

    if approximate_grid != 1:
        # linearly interpolate the zoomed transform grid
        new_x, new_y = np.mgrid[x_min:x_max+1, y_min:y_max+1]
        x_fracs, x_indices = np.modf((x_steps-1)*(new_x-x_min)/float(x_max-x_min))
        y_fracs, y_indices = np.modf((y_steps-1)*(new_y-y_min)/float(y_max-y_min))
        x_indices = x_indices.astype(int)
        y_indices = y_indices.astype(int)
        x1 = 1 - x_fracs
        y1 = 1 - y_fracs
        ix1 = (x_indices+1).clip(0, x_steps-1)
        iy1 = (y_indices+1).clip(0, y_steps-1)
        t00 = transform[0][(x_indices, y_indices)]
        t01 = transform[0][(x_indices, iy1)]
        t10 = transform[0][(ix1, y_indices)]
        t11 = transform[0][(ix1, iy1)]
        transform_x = t00*x1*y1 + t01*x1*y_fracs + t10*x_fracs*y1 + t11*x_fracs*y_fracs
        t00 = transform[1][(x_indices, y_indices)]
        t01 = transform[1][(x_indices, iy1)]
        t10 = transform[1][(ix1, y_indices)]
        t11 = transform[1][(ix1, iy1)]
        transform_y = t00*x1*y1 + t01*x1*y_fracs + t10*x_fracs*y1 + t11*x_fracs*y_fracs
        transform = [transform_x, transform_y]
    return transform

def reposition_ebsd(EBSD_img,
                    microscopy_img,
                    ebsd_correspondence_pts, 
                    dic_correspondence_pts,
                    transform=None, 
                    interp=cv2.INTER_NEAREST):
    """
    Computes and applies Thin Plate Splines (TPS) transform from EBSD raw coordinate system to DIC coordinate system
    
    Arguments
    ---------
    EBSD_img: np.ndarray, 2D
        EBSD field to reposition
    microscopy_img: np.ndarray, 2D
        Microscopy referenced image
    ebsd_correspondence_pts : np.ndarray, 2D, shape=(pts_number, 2) or None
        (y,x) position of correspondence points in EBSD coordinate system
    dic_correspondence_pts : np.ndarray, 2D, shape=(pts_number, 2) or None
        (y,x) position of correspondence points in DIC coordinate system
    transform: np.ndarray, 3D, shape=(2, h, w) or None
        TPS transform
        h : DIC sensor height
        w : DIC sensor width
        If not specified, all previous parameters must be specified and transform is computed
    
    Returns
    -------
    dic_ebsd : np.ndarray, 2D/3D, shape=(h, w, *)
        EBSD field in DIC coordinate system
    transform : np.ndarray, 3D, shape=(2, h, w)
        TPS transform
    """
    h, w = microscopy_img.shape
    if transform is None:
        if all([ebsd_correspondence_pts is not None,
                dic_correspondence_pts is not None,
                h is not None, w is not None]):
            transform_ = _make_inverse_warp(ebsd_correspondence_pts, dic_correspondence_pts, 
                                            [0, 0, h, w], approximate_grid=1)
            transform_ = np.asarray(transform_, dtype=np.float32)
        else:
            raise Exception('Cannot compute transform without further information')
    
    dic_ebsd = cv2.remap(EBSD_img, transform_[1], transform_[0], interp)
    
    if transform is None:
        return dic_ebsd, transform_
    else:
        return dic_ebsd


if __name__ == '__main__' :
    order = [19, 20]
    # Open microscopy_skeleton as referenced image to project all maps
    microscopy_skeleton = cv2.imread('Microscopy_skeleton.png', 0)
    nx, ny = microscopy_skeleton.shape
    # Folder of txt points coordinates
    saving_folder = '2_EBSD_microscopy_txt/'
    
    # List the files path
    datas = []
    for map_number in order :
        datas.append('export CTF/Map' + str(map_number) + '.csv')
    
    # For each EBSD column data, create the .npy file
    column = ['bands', 'error', 'euler1', 'euler2', 'euler3', 'mad', 'bc', 'bs']
    for name in column : 
        print('Processing on ' + name)
        npname = saving_folder+'EBSD_'+name+'.npy'
        if os.path.exists(npname) :
            print('     Loading datas')
            EBSD_full_map_repositionned = np.load(npname)
            ()
        else :
            EBSD_full_map_repositionned = np.zeros((nx,ny))
            for i, EBSD_data in enumerate(datas):
                print('     Map ', i, '/',len(order) )
                # Prepare EBSD original map
                EBSD_img = data_field(EBSD_data, name)
                
                # Load common points between EBSD map and microscopy_skeleton
                EBSD_correspondence_pts = load_correspondence_pts(saving_folder+'Map'+str(order[i])+'.txt')
                microscopy_correspondence_pts = load_correspondence_pts(saving_folder+'Microscopy_Map'+str(order[i])+'.txt')
                
                # Project EBSD on microscopy_skeleton referential
                EBSD_repositionned = reposition_ebsd(EBSD_img,
                                                     microscopy_skeleton,
                                                     EBSD_correspondence_pts, 
                                                     microscopy_correspondence_pts)
            
                # Delete first and last data lines and columns
                EBSD_repositionned = clear_border(EBSD_repositionned[0])
                
                # Add the EBSD map to the general big EBSD one
                EBSD_full_map_repositionned = nd_addition(EBSD_full_map_repositionned,
                                                          EBSD_repositionned)
            
            # Save the full map
            np.save(npname, EBSD_full_map_repositionned)
            # sys.exit()

    # Import all datas and chose a resize parameter to reduce the size of EBSD.ctf file (2000px max is good for Matlab post processing)
    nx, ny = EBSD_full_map_repositionned.shape
    resize_factor = 1
    
    npnames = []
    npfiles = []
    for name in column :
        npname = saving_folder+'EBSD_'+name+'.npy'
        npnames.append(name)
        npfiles.append(cv2.resize(np.load(npname),(int(ny//resize_factor), int(nx//resize_factor))))
    
    # Save all EBSD, repositionned datas into a BIG repositionned EBSD file.
    save_big_file_FCC (npfiles, npnames)