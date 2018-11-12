"""
This file provides 'skeleton' or 'starter' code for MyMD.py. 

Usage: python MyMD.py --iF input.rvc <other params>

Notes: numpy and argparse, particularly the former package, will come in
*extremely* helpful, should you choose to use them, although it is possible 
to complete this assignment without either. 
"""
import argparse
import numpy as np
import csv

class MDSim(object):
    def __init__(self, **kwargs):
        """
        Mothership class for the MD simulation. 
        Parameters: 

        kwargs: from CLI.

        """
        self.read_sim_params = {}  # read from file 
        self.sim_params = {}  # read from cli 
        for key, value in kwargs.items():
            self.sim_params[key] = value
        
       

        self.edges = {} # a dictionary to hold for each atoms the atoms he is linked to.
        self.bonded_atoms = {}
        self.output_row = []
        self.energetics = [['# step',  'E_k' ,'E_b', 'E_nb' ,'E_tot']]
    
    def get_b0(self):
        """
        A method to compute b0 
        As per https://piazza.com/class/jligensd1qn607?cid=403
        we compute the distance between all atoms at t = 0 and use this distnace as b0 for the simulation moving forward
        """
        self.b0 = np.zeros((self.r.shape[1], self.r.shape[1]))
        for i in range(self.r.shape[1]):
            for j in range(self.r.shape[1]):
                index_atom = self.r[:,i]
                target_atom = self.r[:,j]
                d = target_atom-index_atom
                self.b0[i,j] = np.sqrt(np.dot(d,d))



    def parse_rvc_file(self): #, input_file):
        """
        Parses a RVC (positions/velocities/connectivities) file.

        Parameters: 
        -----------
        input_file : name of input RVC file

        Returns: 
        --------
        <This is up to you>
        """
        f = open(self.sim_params['input_file'])
        rows = f.readlines()
        update_parms = rows[0].split(':')[1].strip('\n').split()
        for i in update_parms:
            k,v = i.split('=')
            self.read_sim_params[k] = float(v)
        self.read_sim_params.update(self.sim_params)
        # merge the two dictionaries
        self.sim_params = self.read_sim_params
       


        c = len(rows[1:])+1 #determine number of columns
        #
        self.r = np.zeros((3,c)) # a matrix for position (r_x,r_y,r_z), each col id = atom id
        self.v = np.zeros((3,c)) # matrix to hold velocities Vx,Vy,Vz
        self.F = np.zeros((3,c)) # matrix to hold Fx, Fy, Fz
        self.a = (1/self.sim_params['mass'])*self.F # matrix to hold acceleration vectors
        #
        self.num_atoms = [ i for i in range(1,c)]  # how many atoms are there in the molecule
        self.b = {}  # a dictionary to hold the distances. keyed by atom number and value is a 1 x m matrix.
        #
        self.E_k = np.zeros((1001))
        self.E_b = np.zeros((1001))
        self.E_nb = np.zeros(1001)
        self.E_tot = np.zeros(1001)
        #
        for l in rows[1:]:
            data = l.split()
            #   
            self.r[:,int(data[0])] = data[1:4]
            self.v[:,int(data[0])] = data[4:7]
            edges = []
            bonded_atoms = []
            for e in np.array(data[7:]).astype('int'):
                edges.append([int(data[0]),e])
                bonded_atoms.append(e)
            self.edges[int(data[0])] = edges
            self.bonded_atoms[int(data[0])] = bonded_atoms
        self.format_header()
        self.update_output_row(0)

    def  get_dist(self,atom):
      """
      A method to return the distances from an index atom  and a mask tellig which atom need to be added to the calculation

      """
      d = self.r-self.r[:,atom].reshape(3,1) # subtract the location col from the matrix for each atom
      dist = np.sqrt((d*d).sum(axis=0))
      dist_copy = dist.copy()
      dist_copy[dist_copy >=self.sim_params['nbCutoff']] = 0
      dist_copy[self.bonded_atoms[atom]] = dist[self.bonded_atoms[atom]]
      mask = dist_copy>0
      mask[0] = False  # take care of the zero position
      return dist, mask

    def set_ks(self,atom):
        """
        A method to set a vector of k across all atoms
        will return a vector of [kb, kn ....] for a given atom
        input: inex atom  #atom
        """
        ks = np.ones(self.r.shape[1]) * self.sim_params['kN'] # fill the  vector with kN
        ks[self.bonded_atoms[atom]] = self.sim_params['kB']   # update wherever there is a bonded atom
        # print(ks)
        return ks
    
    def update_force(self,atom):
      """
      A method to return the sum of forces acting on an index atom
      and return a summary  in x,y,z of F
      input index atom 
      retunr a vector of Fx, Fy, Fz
      """
      dist,mask = self.get_dist(atom)

      self.F[:,atom] = np.sum(self.set_ks(atom)*(dist-self.b0[atom])*mask*self.get_proj(atom),axis=1)

    def get_proj(self,atom):
      """
      A method to decomposd the angles of x,y,z components:
      input: index atom
      """
      dist, mask = self.get_dist(atom)
      d = self.r-self.r[:,atom].reshape(3,1) 
      return  np.nan_to_num(1/dist)*d*mask

    def update_positions(self,atom):
      """
      A method to update position
      """
      self.r[:,atom] += self.v[:,atom]*self.sim_params['dt'] # use half way updated speed.

    def update_velocities(self,atom):
      """
      A method to update veolocities
      """
      self.v[:,atom] +=0.5* (1/float(self.sim_params['mass']))*self.F[:,atom]*self.sim_params['dt']
      
    
    def update_kinetic_energy(self):
      """
      A method to return the kinetic energy for the system
      """
      return np.sum(0.5*self.sim_params['mass']*((self.v*self.v).sum(axis=0)))

    def update_potential_energy(self):
      """
      A method to output potential and bonded and non bonded energy
      It basically iterates of the atoms  and sums up the potential energy between bonded and non bonded pairs.
      """
      E_b = 0
      E_nb = 0
      for atom in self.num_atoms:
        dist, mask = self.get_dist(atom)
        ks_vec = self.set_ks(atom)
        kb_mask = ks_vec==self.sim_params['kB']
        knb_mask = ks_vec==self.sim_params['kN']

        E_b += np.sum(0.5*ks_vec*(dist-self.b0[atom])*(dist-self.b0[atom])*mask*kb_mask)
        E_nb += np.sum(0.5*ks_vec*(dist-self.b0[atom])*(dist-self.b0[atom])*mask*knb_mask)
      
      return E_b*0.5 ,E_nb*0.5

    def update_total_energy(self):
      self.E_tot = self.E_k+self.E_nb+self.E_b

    def do_verlet_iteration(self, step):
      """
      A method to carry out the verlet iteration
      """
      for atom in self.num_atoms:
        self.update_velocities(atom)
      
      for atom in self.num_atoms:
        self.update_positions(atom)

      for atom in self.num_atoms:
        self.update_force(atom)

      for atom in self.num_atoms:
        self.update_velocities(atom)
      
      self.E_k[step] = self.update_kinetic_energy()
      self.E_b[step], self.E_nb[step] = self.update_potential_energy()
      self.update_total_energy()


    def format_float(self):
      """
      A method to return a matrix ready for plotting with 4 digits after the decimal point
      input: none
      return: formatted location and velocity matrices 
      """
      shape = self.v.shape
      V = self.v.copy()
      R = self.r.copy()
      v = np.array([format(i,'.4f') for i in V.flatten()]).astype('float').reshape(shape)
      r = np.array([format(i,'.4f') for i in R.flatten()]).astype('float').reshape(shape)
      return r,v


    def update_output_row(self,step):
      """
      A method to update the output row that will be outputed to the file
      input: step  in the iteration.
      """
      r,v = self.format_float()
      data_row = [l.tolist() for l in np.concatenate((r,v),axis=0)[:,1:].T]
      for i,r in enumerate(data_row):
        r.insert(0,i+1)
        r.extend(self.bonded_atoms[i+1])
        self.output_row.append(r)

    def format_text_row(self,step):
      """
      A method to  format the  text update line in the output file
      """ 
      text_row = "#At time step %d,energy = %1.3fkJ"% (step,self.E_tot[step])
      self.output_row.append([text_row])

    def format_header(self):
      """
      A method to  insert a header line in the files
      """
      header = "# %s: kB=%.1f kN=%.1f  nbCutoff=%.2f dt=%.4f  mass=%.2f  T=%.2f" % (self.sim_params['input_file'].split('.')[0], self.sim_params['kB'], self.sim_params['kN'],self.sim_params['nbCutoff'], self.sim_params['dt'], self.sim_params['mass'],self.sim_params['T'])

      self.output_row.append([header])

    def e2float(self,energy):
      """
      A method to format the Energy to one decimal point
      input: energy as a float
      return: return a properly formatted float number
      """
      return float(format(energy,'.1f'))

    def update_energy_trace(self,step):
      """
      A method to update the  Energy tracing
      """
      self.energetics.append([step,self.e2float(self.E_k[step]),self.e2float(self.E_b[step]), self.e2float(self.E_nb[step]), self.e2float(self.E_tot[step])])


    def write_rvc_output(self): #, iter_num): #, file_handle):
        """
        Writes current atom positions and velocities, as well as the iteration
        number, to an output file given by file_handle.

        Parameters: 
        -----------
        iter_num : int, current iteration number.
        file_handle : handle to *_out.rvc file opened for writing
                      (*NOT* file name)

        Returns:
        --------
        None        
        """
        tab_output = self.sim_params['out']+'.rvc'
        with open(tab_output, 'w') as f:
          writer = csv.writer(f, delimiter = '\t')
          writer.writerows(md.output_row)

    def write_erg_output(self): #, iter_num, file_handle):
        """
        Writes energy statistics (kinetic energy, potential energy of 
        bonded interactions, potential energy of nonbonded interactions,
        and the sum of the foregoing energies - E_tot) as well as the iteration
        number to an output file given by file_handle.

        Parameters: 
        -----------
        iter_num : int, current iteration number.
        file_handle : handle to *_out.erg file opened for writing 
                     (*NOT* file name)

        Returns:
        --------
        None
        """
        tab_energy = self.sim_params['out']+'.erg'
        with open(tab_energy, 'w') as f:
          writer = csv.writer(f, delimiter = '\t')
          writer.writerows(md.energetics)

    def run_md_sim(self):
        """
        Runs the MD simulation. 
        """
        time_stamps = [ i for i in range(10,self.sim_params['n']+10,10)]
        for i in range(1,self.sim_params['n']+1):
          self.do_verlet_iteration(i)
          if i in time_stamps:
            self.format_text_row(i)
            self.update_output_row(i)
            self.update_energy_trace(i)

        self.write_rvc_output() 
        self.write_erg_output()   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set of params for p3 simulation')
    parser.add_argument('--iF', 
                    action="store", dest = 'input_file', 
                    required = True,
                    help  = 'input fule for simulation, specify an input file for simulation with suffix .rvc' ,default=False)
        
    parser.add_argument('--kB',   
                        action="store", 
                        dest="kB", default = 40000.0, 
                        type = float,
                        help = 'spring constant for bond')
    parser.add_argument('--kN', action="store", 
                        dest="kN", 
                        default = 400.0,
                        type=float,
                        help = 'spring constant for non bonds')

    parser.add_argument('--nbCutoff', action="store",
                        dest = 'nbCutoff',
                        default=0.5,
                        type = float, 
                        help = 'distance within which atoms should be considered as having nonbonded interactions, if they are not a bonded pair'
                        )
    parser.add_argument('--m', 
                        action = 'store', 
                        dest = 'mass',
                        default = 12.0,
                        type = float,
                        help = 'atom mass, a constant to be applied to all atoms'
    )

    parser.add_argument('--dt', action="store",
                        dest="dt", type=float, default = 0.001,
                        help = 'length of time step')     

    parser.add_argument('--n',action = 'store',
                        dest = 'n',type = int,default = 1000,
                        help =  'number of time steps to iterate, useful for debugging'   )

    parser.add_argument('--out',  
                        action = 'store',
                        default = None,
                        dest = 'out',
                        help = 'name of output file, the suffix _out will be added to the file  the suffixes .rvc and .erg will be used. If no out file specified the name will be taken from  --iF flag and processed.'

                )     

    res = parser.parse_args()
    #res = parser.parse_args(['--iF','square.rvc','--nbCutoff', '3.8'])

    if res.out is None: 
        res.out = res.input_file.split('.')[0]+'_out'
    else:
        res.out = res.out+'_out'
    
   
    kwargs = {k:v  for k,v in res._get_kwargs()}
    md = MDSim(**kwargs)
    #
    md.parse_rvc_file()
    md.get_b0()
    md.run_md_sim()
    
