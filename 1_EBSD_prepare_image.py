import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import pathlib
import os

def transformation_txt (file,x,y): # Remplace tous les termes x par y dans le fichier file
    f = open(file, 'r+')
    text = f.read()
    text = text.strip()
    text = text.lower()
    text = text.replace(x,y)
    wr = open(file, 'w')
    wr.write(text)
            
def data_filter (data) :
    '''
    Tranform all space to ','. Then delete all ',' at the beginning of rows
    '''
    for i in range (20) :
        transformation_txt (data,(20-i)*' ',',')
    transformation_txt (data,'\n,','\n')

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

if __name__ == '__main__' :
    order = [19, 20]
    name = 'euler1'
    datas = []
    for map_number in order :
        datas.append('export CTF/Map' + str(map_number) + '.csv')

    # Prepare datas
    for data in datas :
        data_filter(data)
    EBSDs_parameter = []
    P = pathlib.Path('1_EBSD_prepare_image')
    if os.path.exists(P) :
        ()
    else :
        pathlib.Path.mkdir(P, parents = True) 
    for i in range(len(order)) :
        EBSDs_parameter.append(data_field(datas[i], name))
        # Save maps images in a folder for the Fiji points picking (Used for stitching later)
        filepath = '1_EBSD_prepare_image/Euler1_map'+str(order[i])+'.png'
        imageio.imwrite(filepath, EBSDs_parameter[i])
    