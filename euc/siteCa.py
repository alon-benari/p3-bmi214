import os
import re
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

class SiteCa:
    def __init__(self,atoms):
        """
        A method to initialize the atoms list
        input: atoms list of two atoms
        """
        self.atoms = atoms
    def get_rvc(self):
        """
        A method to return the file names to be processes
        """
        self.file_list = {f.split('_')[1]:f for f in os.listdir() if f.split('.')[1] =='rvc'}
        
    
    def get_rows(self,fname,atom):
        """
        read the files and parse out  the  correct rows"
        and return a c*100 matrix
        the colum number is the atom number - 1
        """
        self.reg = '^'+''.join(['['+i+']' for  i  in str(atom)])+'\s'
        print(self.reg)
        mat = []
        ff = open(fname)
        for row in ff:
            if (re.match(self.reg,row)):
                mat.append(row)

        return  np.array([[ np.float(i) for i in (m.strip('\n' ).split('\t'))] for m in mat])[:,1:4].T
        
    def get_dist(self,fname):
        """
        return a vector of distances in a dictionary
        """
        m0 = self.get_rows(fname, self.atoms[0])
        m1 = self.get_rows(fname, self.atoms[1])
        d = np.sqrt(((m0-m1)*(m0-m1)).sum(axis = 0))
        return d

    def get_plot(self,euc,fname):
        """
        A method to return a plot and save the data as a file
        """
        df = DataFrame.from_dict(euc)
        
        df.set_index(df.index*10)
        df.to_csv(fname)
        #
        fig, ax = plt.subplots(1)
        ax.set_xlabel('steps x10')
        ax.set_ylabel('distance')
        ax.set_title(fname)
        ax.set_xlim([0,1000])
        ax.set_xticks([i for i in range(0,1010,10)])
        df.plot(ax=ax)
        plt.savefig(fname+'.pdf')
        plt.show()
        
        return df
        
        
        
    
### run ####

sc = SiteCa([17,96])
sc.get_rvc()
euc = {}
for k,v in sc.file_list.items():
    euc[k] = sc.get_dist(v)
sc.get_plot(euc,'siteCa1.euc')

sc = SiteCa([298,385])
sc.get_rvc()
euc = {}
for k,v in sc.file_list.items():
    euc[k] = sc.get_dist(v)
sc.get_plot(euc,'siteCa2.euc')





        
