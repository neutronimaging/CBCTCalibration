# this function implements the MCP overlapcorrection as described in: A. S. Tremsin, J. V. Vallerga, J. B. McPhate, and O. H.  # W. Siegmund, Optimization of Timepix count rate capabilities for the applications with a periodic input signal, J. Instrum., vol. 9, no. 5, 2014. https://doi.org/10.1088/1748-0221/9/05/C05026


import glob,sys,os
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import numpy as np
from astropy.io import fits
import shutil

def interAssig(value,intervalDict,df_shutterCount, tolerance): 
    #auxiliary function for OverLapCorrection
    for keyInt in intervalDict:
        if (intervalDict[keyInt][0]-tolerance)<= value <=(intervalDict[keyInt][1]+tolerance):
            return keyInt,df_shutterCount['Counts'][keyInt]

def OverLapCorrection(folder_input, folder_output, filename_output, num_windows):
    
    # here fits and txt files are sorted, the last fits is excluded being the SumImg
    sorted_fits= sorted(glob.glob(folder_input+'/*.fits'))[:-1]
    sorted_TXT= sorted(glob.glob(folder_input+'/*.txt'))
    #display(sorted_TXT)

    # the output folder is created if non-existing
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

#     display(sorted_TXT[0]) #shutter counts
#     display(sorted_TXT[1]) # shutter times
#     display(sorted_TXT[2]) #spectra
#     display(sorted_TXT[3]) # status
    filename_spectra= sorted_TXT[2]
    filename_shuttercount = sorted_TXT[0]
    filename_shuttertime = sorted_TXT[1]
    df = pd.read_csv( filename_spectra,delim_whitespace=True,header=None,
                names=['Time','Spectra'])
    df['diff']=df['Time'].diff()
    df['Spectra']= df['Spectra']/df['Spectra'].max()
    df['diff']= df['diff']/df['diff'].max()

    df_shutterTime = pd.read_csv(filename_shuttertime,delim_whitespace=True,header=None,
                                 names=['t0','t1'], nrows=num_windows)

    df_shutterCount = pd.read_csv(filename_shuttercount,delim_whitespace=True,header=None,
                                 names=['Counts'], nrows=num_windows, index_col=0)

#     display(df_shutterCount)


    df_shutterTime=np.asarray(df_shutterTime.stack(level=[0]))

#     display(df_shutterTime)

    df.plot(x='Time',y=['Spectra','diff'],grid=True,figsize=(8,6))
    sumTime = 0
    TimeArray = np.zeros(num_windows*2)
    index=0
    for i in df_shutterTime:
        print(i)
        sumTime += i
        TimeArray[index] = sumTime
        index += 1
        plt.axvline(x=sumTime)


    plt.show()
#     display(TimeArray)
    interval = {int(key):(value1,value2) for key,value1,value2 in zip(range(len(TimeArray)),TimeArray[::2],TimeArray[1::2])}
    tolerance= (TimeArray[2]-TimeArray[1])/2 #to lower down
    
    dfName = pd.DataFrame(sorted_fits,columns=['name'])
    dfName['ToF'] = df['Time']
    dfName['ShutterWindow']= dfName['ToF'].apply(lambda i:interAssig(i,interval,df_shutterCount, tolerance))
    indexname=0
    
    

    
    i_wb =0 # index used to initialize white beam image

    for names in dfName.groupby('ShutterWindow'):
  
        i=0;
        for idx, value in names[1]['name'].iteritems():
            with fits.open(value) as f:
                array = (f[0].data) # load the image file, size is rows x columns (es. 512x512)
                
                if i==0:
                    sumim=np.zeros(np.shape(array)) # create a blank sum img for each new window
                    prev_array=np.zeros(np.shape(array)) # variable containing the previous array, initialized to zeros (P. Boillat 12.7.19)
                    i += 1 # (P. Boillat 12.7.19)
                                    
#                 sumim+=array # pixel wise sum of loaded imgs
                sumim += (array.astype(float)/2 + prev_array.astype(float)/2) # pixel wise sum of loaded imgs (modified P. Boillat 12.7.19)
                
                if i_wb==0:
                    white_beam=np.zeros(np.shape(array))
                    i_wb = 1
                
#                 if i==0:
#                     P=0
#                     i+=1          
#                 else:
                    P=sumim/names[0][1] # each pixel divided by the shutter count

                newim = array/(1-P) # image correction
                newim= newim.astype(float) # data casting, somehow the convertion to int does not work properly, I have then an error of wrong pixel depth
                white_beam += newim
                filename=filename_output+str(indexname).zfill(5)
    #            display(filename)
                indexname+=1
                fits.writeto(folder_output+filename+'.fits',newim)

#    print(folder_output +  filename_output + 'SummedImg.fits')
    fits.writeto(folder_output +  filename_output + 'SummedImg.fits', white_beam)

# finally I copy the txt files because they can be usefull for the future
    for txt in sorted_TXT:
        filename = txt
        destname = folder_output
        shutil.copy(filename, destname)  
              
    
    
    