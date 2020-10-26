# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:16:23 2019

@author: gathu
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mdlrulelist.util.results2folder import makefolder_time

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

 
datasetnames= ["sonar","haberman","breastCancer","australian","TicTacToe","german",\
               "chess","mushrooms","magic","adult","iris","balance","CMC","page-blocks",\
               "nursery","automobile","glass","dermatology","kr-vs-k","abalone"] 

datasetnames= ["sonar",\
               "german","magic","adult",\
               "balance","kr-vs-k"] 
filesfolder = "./results/all_beam_width/"
results = dict()
df = pd.read_csv(filesfolder+"summary.csv")
for datasetname in datasetnames:
    dfaux= df[df["datasetname"]==datasetname]
    results[datasetname] = dict()
    results[datasetname]["beamsize"]= dfaux.index.values
    results[datasetname]["length_ratio"]  = np.round (dfaux.length_ratio.values,2)  
    results[datasetname]["wkl_sum"] = dfaux.wkl_sum.values 
    results[datasetname]["time"] = dfaux.runtime
    
# now make a plot!!!"#!"#!"#!"#!"#!"#!#!
def make_graph(results,x_str,y_str,size_marker,color=tableau20):
    alp = 1
    fig= plt.figure()
    #fig = plt.gca()
    for iname,name in enumerate(results):  
        x = results[name][x_str]
        y = results[name][y_str]
        #plt.semilogx(x, y,alpha =alp,c=np.array(color[iname])/255, marker='o',label=name,\
        #        linewidth = 0.5,markersize = size_marker)
        plt.semilogx(x, y,alpha =alp,c=np.array(color[2*iname])/255, marker='o',label=name,\
                linewidth = 0.5,markersize = size_marker)
    #plt.ticklabel_format(style='plain')   
    plt.grid(b=True, which='major', axis='y', linestyle= '--', linewidth=0.6)
    lgd =plt.legend(loc='upper right')
    return fig,lgd
folder_path = makefolder_time()
fig,lgd = make_graph(results,x_str="beamsize",y_str="length_ratio",size_marker = 6)
plt.xlabel("beam's width")
#plt.ticklabel_format(style='plain', axis='y')
plt.ylabel("relative compression")
fig.savefig(os.path.join(folder_path,"beamwidth_compression.pdf"), bbox_extra_artists=(lgd,), bbox_inches='tight')

folder_path = makefolder_time()
fig,lgd = make_graph(results,x_str="beamsize",y_str="time",size_marker = 6)
plt.xlabel("beam's width")
plt.ylabel("time (seconds)")
fig.savefig(os.path.join(folder_path,"beamwidth_runtime.pdf"), bbox_extra_artists=(lgd,), bbox_inches='tight')    



results_runtime = np.array([[6.5635,np.nan,299.3277,231.4856,np.nan],
[0.2344,0.2813,5.3129,6.344,20.065],
[2.6777,16.8081,18.8102,22.0132,11.8312],
[4.926,321.7971,47.3165,137.1699,13.5494],
[1.9533,0.3907,41.8473,80.4448,np.nan],
[3.344,np.nan,209.1891,466.0406,np.nan],
[4.0003,1613.2774,192.1963,925.5005,np.nan],
[3.6891,6.3441,736.4174,142.0154,np.nan],
[30.4253,np.nan,5007.5807,26552.5521,np.nan],
[84.4331,np.nan,np.nan,593.4482,np.nan],
[0.3518,np.nan,2.5783,1.7584,14.8001],
[1.0782,np.nan,8.1633,np.nan,20.708],
[6.511,np.nan,77.04,119.4047,11.9107],
[9.725,np.nan,731.2944,992.5822,21.9084],
[8.1114,np.nan,273.4512,478.9526,np.nan],
[6.0324,np.nan,38.6677,37.3615,np.nan],
[2.4846,np.nan,10.3533,7.7469,18.3],
[14.1672,np.nan,33.9093,45.9088,10.0611],
[121.3676,np.nan,1153.4578,9846.1516,np.nan],
[13.6274,np.nan,445.4329,2336.3557,np.nan]])

s=25
alp = 0.9
fig = plt.figure()
ax = plt.gca()
list_markers=['s','D','v','^','<',"o",'>']
algorithms = ["SSD++","FSSD","DSSD","CN2SD","MCTS4DM"]
my_xticks =["sonar","haberman","breast","australian","TicTacToe","german",\
               "chess","mushrooms","magic","adult","iris","balance","CMC","page-blocks",\
               "nursery","automobile","glass","dermatology","kr-vs-k","abalone"] 
x = np.array([i for i in range(1,len(my_xticks)+1)])
ax.axvline(10.5,linewidth =1,linestyle="-.", color =(0,0,0))
for ialg,alg in enumerate(algorithms):
    ax.scatter(x, results_runtime[:,ialg],s,alpha =alp,
               c=np.array(tableau20[2*ialg])/255,edgecolor = (0,0,0),
               marker=list_markers[ialg],label=alg)
    
#ax.axvline(9.5,linewidth =1,linestyle="-.", color =(0,0,0))

#plt.ylim( (0.01, 1000) ) 
#ax.yaxis.grid(True)
ax.grid(b=True, which='major', axis='y', linestyle= '--', linewidth=0.6)

ax.set_yscale('log')
ax.set_xticks( x  )

ax.set_xticklabels(my_xticks,fontdict={'fontsize':11,\
                                       'rotation':'45',\
                                       "horizontalalignment":'right'})    
#plt.ylim( (10**-3, 10**3) ) 
#plt.scatter(x, y, marker='^')
#plt.scatter(x, y, s=area2, marker='o', c=c)
#plt.xticks(rotation=60)
plt.xlabel("datasets")
plt.ylabel("runtime (seconds)")
#plt.legend(loc=1)
#plt.legend([plot1])
#lgd =ax2.legend(loc='upper right', bbox_to_anchor=(0.34,1))
folder_path = makefolder_time()
lgd =plt.legend(loc='upper right', bbox_to_anchor=(0.3,1.03))
fig.savefig(os.path.join(folder_path,"algorithms_runtime.pdf"), bbox_extra_artists=(lgd,), bbox_inches='tight')    

#plt.tight_layout()
#plt.show()

results_jaccard= np.array([[0,np.nan,15.12,2.91,np.nan],
[0,0,18.41,8.17,0],
[25.59,0,46.5,12.04,13.8],
[15.15,0,24.51,15.43,7.89],
[2.4,0,8.44,13.76,np.nan],
[6.62,np.nan,9.24,10.33,np.nan],
[12.89,0,11.47,16.52,np.nan],
[17.58,0,9.34,1.99,np.nan],
[2.91,np.nan,17.2,15.21,np.nan],
[1.83,np.nan,np.nan,8.38,np.nan],
[32.05,np.nan,22.26,19.39,4.17],
[10.98,np.nan,16.67,np.nan,8.65],
[5.77,np.nan,22.09,23.4,6.44],
[4.58,np.nan,28.35,19.48,10.98],
[2.6,np.nan,14.13,13.48,np.nan],
[10.6,np.nan,19.46,29.94,np.nan],
[40.25,np.nan,17.74,4.08,6.21],
[14.77,np.nan,11.74,26.01,13.62],
[0.39,np.nan,22.91,14.76,0],
[7.89,np.nan,35.27,48.77,0]])

s=25
alp = 0.9
fig = plt.figure()
ax = plt.gca()
list_markers=['s','D','v','^','<',"o",'>']
algorithms = ["SSD++","FSSD","DSSD","CN2SD","MCTS4DM"]
my_xticks =["sonar","haberman","breast","australian","TicTacToe","german",\
               "chess","mushrooms","magic","adult","iris","balance","CMC","page-blocks",\
               "nursery","automobile","glass","dermatology","kr-vs-k","abalone"] 
x = np.array([i for i in range(1,len(my_xticks)+1)])
ax.axvline(10.5,linewidth =1,linestyle="-.", color =(0,0,0))
for ialg,alg in enumerate(algorithms):
    ax.scatter(x, results_jaccard[:,ialg],s,alpha =alp,
               c=np.array(tableau20[2*ialg])/255,edgecolor = (0,0,0),
               marker=list_markers[ialg],label=alg)
    
#ax.axvline(9.5,linewidth =1,linestyle="-.", color =(0,0,0))

#plt.ylim( (0.01, 1000) ) 
#ax.yaxis.grid(True)
ax.grid(b=True, which='major', axis='y', linestyle= '--', linewidth=0.6)

ax.set_xticks( x  )

ax.set_xticklabels(my_xticks,fontdict={'fontsize':11,\
                                       'rotation':'45',\
                                       "horizontalalignment":'right'})    
#plt.ylim( (100, 0) ) 
#plt.scatter(x, y, marker='^')
#plt.scatter(x, y, s=area2, marker='o', c=c)
#plt.xticks(rotation=60)
plt.xlabel("datasets")
plt.ylabel("jaccard index average (%)")
#plt.legend(loc=1)
#plt.legend([plot1])
#lgd =ax2.legend(loc='upper right', bbox_to_anchor=(0.34,1))
folder_path = makefolder_time()
lgd =plt.legend(loc='upper right', bbox_to_anchor=(0.45,1.03))
fig.savefig(os.path.join(folder_path,"algorithms_jaccard.pdf"), bbox_extra_artists=(lgd,), bbox_inches='tight')    

