####################################################################################################
######################### RuNNer universal converter ###############################################
####################################################################################################
'''
This is the RuNNer universal converter RuNNerUC. Type in RuNNerUC -h to find how to use it or look
at the "help1" string.

If you want to add a new format to the RuNNerUC, there are only a few things to look for:
For reading: The output of the read functions must consist of equally sized arrays for comments,
energy, charge, lattice, and atoms in this order. The input must take the infile and the options.
The charge array consists of charges and the setting if charge was found (True) or if the default
charge was used (False).
For writing: The function needs to take in comments, energy, charge, lattice, atoms, outfile,
defined_frame, and options in this order. Add warning if force of atoms or energy or charge of
structure was set to the default value! There is a function to do so!

Steps to add a new reader/writer:
1) Add the function name to the dict_read/dict_write arrays towards the bottom of the script. Example:
def myconvfunction():
    pass
dict_read= {'runner':read_input_data,  'lammpstrj':read_lammpstrj,  'aims_geo':read_geometry_in, 'myformat':myconvfunction}

2) Read functions: The callsign for read functions is simple:
def myreadfunc(infile,options):
  return(comments, energy, charge, lattice, atoms)

Where infile is a string with the name of a file, and options is a list of the form ("myoption1=periodic", "myoption2=vdw", etc.).
You can use the helper function get_options to parse this options and their value. 
The file should be read as a whole, so if it has multiple frames/is a trajectory, you need to read and return all the frames' data
The data should be returned in RuNNer units (Distances: Bohr; Energy: Ha; Forces: Ha/Bohr; Charges: electron charge)
The data is to be returned as arrays: 
    comments is an array of arrays of strings per frame. If there is only one comment in your file (or none at all), please reutrn as [["hello",], ["world",], etc] to create one object lists
    energy, charge are arrays of one float value per frame
    atoms is an array of arrays of atom objects. each frame needs an array of atom objects, with as many atom objects as real atoms in the frame (check the atom class definition for more about the object)
    lattice is an array of three lattice vectors in the following format: (don't expect it to be a numpy array)
        [[a1x, a1y, a1z],
         [a2x, a2y, a2z],
         [a3x, a3y, a3z],]

If the format you are reading does not define a property, please return the following defaults:
    comments=[[]]
    energy=[99.999999,]
    charge=[[0.0,False],]
    lattice=[None]
    atomic properties (force, charge, energy): not required, the atoms object already defines its own defaults

Remember to close your read file!

3) Write functions: The call sign is a bit more complex:
def mywritefunc(comments,energy,charge,lattice,atoms,outfile,frame,options):
    return

There is no return value in this case.
The input parameters are as described in the previous point, mostly lists/arrays. 
The new object is the frame parameter, which is a list of chosen frames indexed at 0 (better described in define_frame function).
This allows you to simply index all the other arrays based on the integers inside the frame list.
You should assume the data is being passed to you in RuNNer units.
There is a function to add comments in case your data has some default values. Call it as:
for i in frame:
    comments[i]=check_for_default(i,energy[i],charge[i],atoms[i],comments[i],options)
comments[i] will be appended with an extra comment string if a default value was detected

Remember to close your write file!

4) PLEASE keep your functions organized in pairs in the code, first the read and then the write functions for the same format!
read_xformat:
    pass
write_xformat:
    pass
read_yformat:
    pass
write_yformat:
    pass
etc.

Of course this doesn't apply to some files that should never be read or written (ex: VASP's OUTCAR format is read only). In those cases, try to keep related formats close together (all VASP formats, all LAMMPS formats, etc.).

5) You should detect incompatibilities for your format (for example, if you are asked to write more than one frame for a format that can only ever have one frame). 
   It's preferrable to exit entirely in such case, since warnings can be missed. Print a warning and exit:
    print("WARNING/ERROR: Option X is incompatible with format Y, exiting")
    sys.exit(2) #the 2 is an error code that can be captured by bash for example

6) For your float output, please use the 12.6f format

Things that have to be done:
Add help commands that show all available options for given formats
'''

####################################################################################################
########################################## Dependencies ############################################
####################################################################################################

import sys 
import numpy as np
import getopt

####################################################################################################
########################################## Units and definitions ###################################
####################################################################################################

help1='RuNNerUC.py <format_reading> <format_writing> -i <inputfile> -o <outputfile> -c "<comment>" \
-f <frame> <options>\n options without -X in front have to be given after all other options in\
 the format option=value_of_option \n To get a list of all available formats, use python RuNNerUC -l'

Bohr2Ang = 0.5291772109030   # CODATA 2018
Ang2Bohr = 1/Bohr2Ang
Eh2eV    = 27.211386245988   # CODATA 2018
eV2Eh    = 1/Eh2eV

element_masses={#First Period
                'H' :1.0079,   'He':4.0026,
                #Second Period
                'Li':6.941,    'Be':9.0122,   'B' :10.811,   'C' :12.0107,  'N' :14.0067,  'O' :15.9994,  'F' :18.9984,  'Ne':20.1797,
                #Third Period
                'Na':22.9897,  'Mg':24.305,   'Al':26.9815,  'Si':28.0855,  'P' :30.9738,  'S' :32.065,   'Cl':35.453,   'Ar':39.948, 
                #Fourth Period
                'K' :39.0983,  'Ca':40.078,   'Sc':44.9559,  'Ti':47.867,   'V' :50.9415,  'Cr':51.9961,  'Mn':54.938,   'Fe':55.845,  'Co':58.9332,
                'Ni':58.6934,  'Cu':63.546,   'Zn':65.39,    'Ga':69.723,   'Ge':72.64,    'As':74.9216,  'Se':78.96,    'Br':79.904,  'Kr':83.8, 
                #Fifth Period
                'Rb':85.4678,  'Sr':87.62,    'Y' :88.9059,  'Zr':91.224,   'Nb':92.9064,  'Mo':95.94,    'Tc':98,       'Ru':101.07,  'Rh':102.9055, 
                'Pd':106.42,   'Ag':107.8682, 'Cd':112.411,  'In':114.818,  'Sn':118.71,   'Sb':121.76,   'Te':127.6,    'I':126.9045, 'Xe':131.293, 
                #Sixth Period
                'Cs':132.9055, 'Ba':137.327,  'La':138.9055, 'Ce':140.116,  'Pr':140.9077, 'Nd':144.24,   'Pm':145,      'Sm':150.36,  'Eu':151.964,  
                'Gd':157.25,   'Tb':158.9253, 'Dy':162.5,    'Ho':164.9303, 'Er':167.259,  'Tm':168.9342, 'Yb':173.04,   'Lu':174.967, 'Hf':178.49, 
                'Ta':180.9479, 'W':183.84,    'Re':186.207,  'Os':190.23,   'Ir':192.217,  'Pt':195.078,  'Au':196.9665, 'Hg':200.59,  'Tl':204.3833, 
                'Pb':207.2,    'Bi':208.9804, 'Po':209,      'At':210,      'Rn':222,
                #Seventh Period
                'Fr':223,      'Ra':226,      'Ac':227,      'Th':232.0381, 'Pa':231.0359, 'U':238.0289,  'Np':237,      'Pu':244,     'Am':243, 
                'Cm':247,      'Bk':247,      'Cf':251,      'Es':252,      'Fm':257,      'Md':258,      'No':259,      'Lr':262,     'Rf':261, 
                'Db':262,      'Sg':266,      'Bh':264,      'Hs':277,      'Mt':268}

#Need a pre-sorted version of this:
elements_sorted=[s[0] for s in sorted(element_masses.items(), key=lambda key_values:key_values[1])] #['H', 'He', ..., 'Hs', 'Mt']

thres=0.000000001

####################################################################################################
########################################## Functions ###############################################
####################################################################################################

def main(argv):
    '''
    Read in the command line arguments.  
    '''
    inputfile = None
    outputfile = None
    frame = 'All'
    given_comments = None
    options = []
    try:
        opts, args = getopt.getopt(argv,"h:i:o:f:c:")
    except getopt.GetoptError:
        print(help1)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help1)
            print('available writing formats:')
            for i in dict_write:
                print(i, dict_write[i].__doc__)
            print('available reading formats:')
            for i in dict_read:
                print(i, dict_read[i].__doc__)
            sys.exit()
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-o"):
            outputfile = arg
        elif opt in ("-f"):
            frame = arg
        elif opt in ("-c"):
            given_comments = arg
    options = args  
    for i in options:
        if i[0]=='-':
            print(help1)
            sys.exit(2)

    return(inputfile,outputfile,frame,given_comments,options)    

####################################################################################################

class atom():
    '''
    Atom class, used to store the information of individual atoms. Use RuNNer units (bohr and Hartree). 
    Use atom_energy as placeholder if needed. Atom_charge and atom energy will be set to 0 if not given. 
    Force will be set to 99.999999 if not given. Will then also set default_f to True
    
    Callsign:
    mynewatom=atom(element='', x=, y=, z=, [fx=, fy=, fz=, atom_charge=, atom_energy=])
    where the values within [] are optional and can be omitted (they get a default)
    Properties:
    -element: Should be the periodic table name of the element as a string
    -xyz:     List/array of the shape [x, y, z]
    -force:   List/array of the shape [fx, fy, fz]
    -atom_charge, atom_energy: Single floats
    '''
    def __init__(self, **kwargs):
        self.default_f=False
        try: 
            self.element=kwargs['element']
            #print(self.element)
        except:
            print('Atom was initialized without element type')
            sys.exit()
        try:
            self.xyz=[kwargs['x'],kwargs['y'],kwargs['z']]
            #print(self.xyz)
        except:
            print('Atom was initialized without xyz')
            sys.exit()
        fx=kwargs.pop('fx',99.999999)
        fy=kwargs.pop('fy',99.999999)
        fz=kwargs.pop('fz',99.999999)
        if abs(fx-99.999999)<thres or abs(fy-99.999999)<thres or abs(fz-99.999999)<thres:
            self.default_f=True
        self.force=[fx,fy,fz]
        atom_charge=kwargs.pop('atom_charge',0.)
        atom_energy=kwargs.pop('atom_energy',0.)
        self.atom_charge=atom_charge
        self.atom_energy=atom_energy       

####################################################################################################

def get_options(options, option):
    option_value=None
    for i in options:
        if i.strip().split('=')[0] == option:
            option_value=i.strip().split('=')[1]
    return option_value

####################################################################################################

def read_input_data(infile,options):
    '''
     Function to read in RuNNer files. If infile is None defaults to input.data. Gives error if atom 
     line in infile has not 10 entries. Has no special Options
    '''
    if infile == None:
        infile='input.data'
    comments=[[]]
    lattice=[None]
    atoms=[[]]
    energy=[99.999999]
    charge=[[0.0,False]]
    structure=0
    got_lattice=False
    with open(infile,'r') as f:
        lines=f.readlines()
    for i in np.arange(0,len(lines)):
        if lines[i].strip()=='begin':
            structure+=1
            comments.append([])
            energy.append(99.999999)
            charge.append([0.0,False])
            atoms.append([])
            lattice.append([None])
            got_lattice=False
        elif lines[i].split()[0].strip()=='atom':
            line=lines[i].strip().split()
            if len(line) != 10:
                print('Atom line {} is incomplete!'.format(i))
                sys.exit()
            atoms[structure].append(atom(element=str(line[4]),x=float(line[1]),y=float(line[2]),z=float(line[3]),fx=float(line[7]),fy=float(line[8]),fz=float(line[9]),atom_energy=float(line[6]),atom_charge=float(line[5]))) 
        elif lines[i].split()[0].strip()=='lattice':
            if got_lattice==False:
                got_lattice=True
                line1=lines[i].strip().split()
                line2=lines[i+1].strip().split()
                line3=lines[i+2].strip().split()
                lattice[structure]=np.array([[float(line1[1]),float(line1[2]),float(line1[3])],[float(line2[1]),float(line2[2]),float(line2[3])],[float(line3[1]),float(line3[2]),float(line3[3])]])
        elif lines[i].split()[0].strip()=='comment':
            comments[structure].append(lines[i][8:].strip())
        elif lines[i].split()[0].strip()=='c':
            comments[structure].append(lines[i][2:].strip())
        elif lines[i].split()[0].strip()=='energy':
            energy[structure]=float(lines[i].strip().split()[1])
        elif lines[i].split()[0].strip()=='charge':
            charge[structure][0]=float(lines[i].strip().split()[1])
            charge[structure][1]=True

    return(comments[1:],energy[1:],charge[1:],lattice[1:],atoms[1:])

####################################################################################################

def read_lammpstrj(infile, options):
  '''
  Function to read in lammpstrj files. The length unit is expected to be Angstrom. The units for the
  charge q, the atomic energy E, and the forces fx, fy, and fz can be specified in the beginning
  of the function. No empty lines are allowed in the lammpstrj file, otherwise the function will
  stop reading at this point. If infile is None, the default file name is input.lammpstrj. The
  output arrays are Python arrays.
  '''
  q_conversion = 1
  E_conversion = 1
  f_conversion = Bohr2Ang/Eh2eV

  properties = ['x', 'y', 'z', 'element', 'q', 'E', 'fx', 'fy', 'fz']

  if infile == None:
    infile = 'input.lammpstrj'

  comments = []
  lattice = []
  atoms = []
  with open(infile) as f:
    line = f.readline().strip()
    while not line.startswith('ITEM: TIMESTEP') and line:
      line = f.readline().strip()
    if line:
      comments.append(['timestep '+f.readline().strip()])
      line = f.readline().strip()
    while line:
      while not line.startswith('ITEM: BOX') and line:
        line = f.readline().strip()
      if line:
        lat = np.array([f.readline().strip().split(), f.readline().strip().split(), f.readline().strip().split()]).astype(float)
        if len(lat[0])==2:
          lattice.append([[(lat[0][1]-lat[0][0])/Bohr2Ang, 0.0, 0.0], [0.0, (lat[1][1]-lat[1][0])/Bohr2Ang, 0.0], [0.0, 0.0, (lat[2][1]-lat[2][0])/Bohr2Ang]])
        elif len(lat[0])==3:
          lx = lat[0][1]-lat[0][0]-np.array([0.0, lat[0][2], lat[1][2], lat[0][2]+lat[1][2]]).max()+np.array([0.0, lat[0][2], lat[1][2], lat[0][2]+lat[1][2]]).min()
          ly = lat[1][1]-lat[1][0]-np.array([0.0, lat[2][2]]).max()+np.array([0.0, lat[2][2]]).min()
          lz = lat[2][1]-lat[2][0]
          lattice.append([[lx/Bohr2Ang, 0.0, 0.0], [lat[0][2]/Bohr2Ang, ly/Bohr2Ang, 0.0], [lat[1][2]/Bohr2Ang, lat[2][2]/Bohr2Ang, lz/Bohr2Ang]])
        else:
          print('Definition of lattice is unknown.')
          exit()
        line = f.readline().strip()
        while not line.startswith('ITEM: ATOMS') and line:
          line = f.readline().strip()
        order = np.array(line.split())[2:]
        order_index = {}
        for prop in properties:
          try:
            order_index[prop] = np.where(order==prop)[0][0]
          except IndexError:
            order_index[prop] = None
        atoms.append([])
        line = f.readline().strip()
        while not line.startswith('ITEM:') and line:
          line = line.split()
          atoms[-1].append(atom(element=line[order_index['element']],x=float(line[order_index['x']])/Bohr2Ang,y=float(line[order_index['y']])/Bohr2Ang,z=float(line[order_index['z']])/Bohr2Ang))
          if order_index['q']!=None:
            atoms[-1][-1].atom_charge = line[order_index['q']]*q_conversion
          if order_index['E']!=None:
            atoms[-1][-1].atom_energy = line[order_index['E']]*E_conversion
          if order_index['fx']!=None:
            atoms[-1][-1].force[0] = float(line[order_index['fx']])*f_conversion
          if order_index['fy']!=None:
            atoms[-1][-1].force[1] = float(line[order_index['fy']])*f_conversion
          if order_index['fz']!=None:
            atoms[-1][-1].force[2] = float(line[order_index['fz']])*f_conversion
          if order_index['fx']!=None and order_index['fy']!=None and order_index['fz']!=None:
            atoms[-1][-1].default_f = False
          line = f.readline().strip()
        while not line.startswith('ITEM: TIMESTEP') and line:
          line = f.readline().strip()
        if line:
          comments.append(['timestep '+f.readline().strip()])
          line = f.readline().strip()

  n_structures = len(lattice)
  energy = [99.999999]*n_structures
  charge = [[0.0,False]]*n_structures

  return(comments, energy, charge, lattice, atoms)

####################################################################################################

def read_geometry_in(infile,options):
    '''
    function to read in geometry.in files used by fhi-aims. if infile is none, the default file name 
    is geometry.in .
    '''
    if infile == None:
        infile = 'geometry.in'
    atoms=[[]]
    comments=[[]]
    energy=[99.999999]
    charge=[[0.0,False]]
    got_lattice=False
    lattice=[[]]
    with open(infile,'r') as f:
        lines=f.readlines()
    for i in np.arange(0,len(lines)):
        if lines[i].split()[0].strip()=='atom':
            line=lines[i].strip().split()
            if len(line) != 5:
                print('atom line {} is incomplete! (or too long)'.format(i))
                sys.exit()
            atoms[0].append(atom(element=str(line[4]),x=float(line[1])/Bohr2Ang,y=float(line[2])/Bohr2Ang,z=float(line[3])/Bohr2Ang)) 
        elif lines[i].split()[0].strip()=='lattice_vector':
            if got_lattice==False:
                got_lattice=True
                line1=lines[i].strip().split()
                line2=lines[i+1].strip().split()
                line3=lines[i+2].strip().split()
                lattice=[np.array([[float(line1[1])/Bohr2Ang,float(line1[2])/Bohr2Ang,float(line1[3])/Bohr2Ang],[float(line2[1])/Bohr2Ang,float(line2[2])/Bohr2Ang,float(line2[3])/Bohr2Ang],[float(line3[1])/Bohr2Ang,float(line3[2])/Bohr2Ang,float(line3[3])/Bohr2Ang]])]
        elif lines[i][0]=='#':
            comments=[[lines[i].strip()]]

    return(comments,energy,charge,lattice,atoms)

####################################################################################################

def read_aims_output(infile1,options1):
    '''
     Function to read in fhi-aims output files (single point an relaxations). If infile is None defaults to data.own. Options are: get_spin to get the spin of the atoms and save them in the atom_energy value of the atom object.
    '''
    f_conversion = eV2Eh/Ang2Bohr
    if infile1 == None:
        infile1='data.own'
    comments=[[]]
    lattice=[[]]
    atoms=[[]]
    energy=[99.999999]
    charge=[[0.0,False]]
    structure=0
    got_lattice=False
    with open(infile1,'r') as f:
        lines=f.readlines()
    get_spin=get_options(options, 'get_spin') 
    start_control_info=None
    start_structure_info=None
    start_final_structure_info=None
    performing_relaxation=False
    performing_MD=False
    start_forces=None
    start_spins=None
    count=0
    n_atoms=-1
    found_energy=False
    start_control_info1=None
    start_structure_info1=None
    start_forces1=None
    start_spins1=None

    for i in np.arange(0,len(lines)):  # first pass to get loactions of data
        if lines[i].strip()=='Reading file control.in.':
            start_control_info=i
        elif lines[i].strip()=='Reading geometry description geometry.in.':
            start_structure_info=i
        elif len(lines[i]) >= 18 and lines[i].strip()[0:20] == 'Geometry relaxation:':
            performing_relaxation=True
        elif lines[i].strip()[0:43] == 'Running Born-Oppenheimer molecular dynamics':
            performing_MD=True 
        elif lines[i].strip()[0:22] == 'Final atomic structure':
            start_final_structure_info=i
        elif lines[i].strip()=='Total atomic forces (unitary forces cleaned) [eV/Ang]:':
            start_forces=i
        elif len(lines[i].strip().split())==6:
            if lines[i].strip().split()[3]=='atoms':
                n_atoms=int(lines[i].strip().split()[5])
        elif len(lines[i].strip().split())>10:
            if lines[i].strip().split()[1]=='Total':
                if lines[i].strip().split()[0]=='|':
                    if lines[i].strip().split()[2]=='energy':
                        if lines[i].strip().split()[3]=='of':
                            energy[structure]=float(lines[i].strip().split()[11])*eV2Eh
                            found_energy=True      
        elif get_spin=='True':
            if lines[i].strip()=='Performing Hirshfeld analysis of fragment charges and moments.':
                start_spins=i
    if found_energy==False:
        print('No energy found!!!')
        exit() 
    
    for i in np.arange(start_control_info,start_control_info+22): #to get all important informations, charge, functional, k grid etc
        if len(lines[i].split())>2:
            if lines[i].split()[0].strip()=='Charge':
                charge[structure]=[float(lines[i].split()[2].strip()[:6]),True]
            if lines[i].split()[0].strip()=='Spin':
                if lines[i].split()[2].strip()=='No':
                    if get_spin==True:
                        get_spin=False
    if not performing_relaxation and not performing_MD:
        for i in np.arange(start_structure_info,len(lines)):
            if len(lines[i].split())>2:
                if lines[i].split()[2].strip()=='Species':
                    line=lines[i].strip().split()
                    atoms[structure].append(atom(element=str(line[3]),x=float(line[4])*Ang2Bohr,y=float(line[5])*Ang2Bohr,z=float(line[6])*Ang2Bohr))
                elif got_lattice==False:
                    if lines[i].split()[1].strip()=='Unit':
                        got_lattice=True
                        line1=lines[i+1].strip().split()
                        line2=lines[i+2].strip().split()
                        line3=lines[i+3].strip().split()
                        lattice[structure]=np.array([[float(line1[1])*Ang2Bohr,float(line1[2])*Ang2Bohr,float(line1[3])*Ang2Bohr],[float(line2[1])*Ang2Bohr,float(line2[2])*Ang2Bohr,float(line2[3])*Ang2Bohr],[float(line3[1])*Ang2Bohr,float(line3[2])*Ang2Bohr,float(line3[3])*Ang2Bohr]])
                    elif lines[i].strip()=='| No unit cell requested.':
                        got_lattice=True
                if len(atoms[0]) >= n_atoms and got_lattice==True:
                    break
    elif performing_relaxation or performing_MD:
        print('WARNING: Found a geometry relaxation or a MD run. Only the last structure will be taken.')
        for i in np.arange(start_final_structure_info,len(lines)):
            if len(lines[i].split())>2:
                if lines[i].split()[0].strip()=='atom':
                    line=lines[i].strip().split()
                    atoms[structure].append(atom(element=str(line[4]),x=float(line[1])*Ang2Bohr,y=float(line[2])*Ang2Bohr,z=float(line[3])*Ang2Bohr))
                elif got_lattice==False:
                    if lines[i].split()[0].strip()=='lattice_vector':
                        got_lattice=True
                        line1=lines[i].strip().split()
                        line2=lines[i+1].strip().split()
                        line3=lines[i+2].strip().split()
                        lattice[structure]=np.array([[float(line1[1])*Ang2Bohr,float(line1[2])*Ang2Bohr,float(line1[3])*Ang2Bohr],[float(line2[1])*Ang2Bohr,float(line2[2])*Ang2Bohr,float(line2[3])*Ang2Bohr],[float(line3[1])*Ang2Bohr,float(line3[2])*Ang2Bohr,float(line3[3])*Ang2Bohr]])
                    elif lines[i].strip()=='| No unit cell requested.':
                        got_lattice=True
                if len(atoms[structure]) >= n_atoms and got_lattice==True:
                    break


    if start_forces != None:
        for i in np.arange(start_forces,start_forces+n_atoms+2):
            if len(lines[i].split())>1:
                if lines[i].split()[0].strip()=='|':
                    count+=1
                    line=lines[i].strip().split()
                    atoms[structure][count-1].force=[float(line[2])*f_conversion,float(line[3])*f_conversion,float(line[4])*f_conversion]
                    atoms[structure][count-1].default_f=False
    count=0
    if get_spin=='True':
        for i in np.arange(start_spins,start_spins+n_atoms*11+4):
            if len(lines[i].split())==6:
                if lines[i].split()[2].strip()=='spin':
                    count+=1
                    atoms[structure][count-1].atom_energy=float(lines[i].split()[5].strip())

    print(len(comments), len(energy), len(charge), len(lattice), len(atoms))

    return(comments,energy,charge,lattice,atoms)

####################################################################################################

def read_aims_traj(infile1,options):
    '''
    Function to read a all structures of a MD trajectory or all steps of a relaxation.
    '''
    f_conversion = eV2Eh/Ang2Bohr
    if infile1 == None:
        infile1='data.own'
    comments=[[]]
    lattice=[[]]
    atoms=[[]]
    energy=[99.999999]
    charge=[[0.0,False]]
    structure=-1
    got_lattice=False
    no_cell_needed = False
    with open(infile1,'r') as f:
        lines=f.readlines()
    start_control_info=None
    start_structure_info=None
    start_frame_structure_info = []
    performing_relaxation=False
    performing_MD=False
    start_forces= []
    start_spins= []
    count=0
    n_atoms=-1
    found_energy=False
    start_control_info1=None
    start_structure_info1=None
    start_forces1=None
    start_spins1=None
    scf_converged = None

    for i in np.arange(0,len(lines)):  # first pass to get loactions of data and of the initial structure
        if lines[i].strip()=='Reading file control.in.':
            start_control_info=i
        elif lines[i].strip()=='Reading geometry description geometry.in.':
            start_structure_info=i
            structure += 1
            start_frame_structure_info.append(i)
            scf_converged = False
            if (len(energy) - 1) < structure:
                                       energy.append(99.999999)
                                       found_energy = False
        elif len(lines[i]) >= 18 and lines[i].strip()[0:20] == 'Geometry relaxation:':
            performing_relaxation=True
        elif lines[i].strip()[0:43] == 'Running Born-Oppenheimer molecular dynamics':
            performing_MD=True 
        elif len(lines[i].strip().split())==6:
            if lines[i].strip().split()[3]=='atoms':
                n_atoms=int(lines[i].strip().split()[5])

        # Reading data of relaxation steps
        if performing_relaxation or performing_MD:
                   if len(lines[i]) > 22 and lines[i].strip()[0:22] == 'Relaxation step number':
                             structure += 1
                             start_frame_structure_info.append(i)
                             scf_converged = False
                             if (len(energy) - 1) < structure:
                                       comments.append([])
                                       lattice.append([])
                                       atoms.append([])
                                       energy.append(99.999999)
                                       charge.append([0.0,False])
                                       found_energy = False
                                       got_lattice=False
                   elif len(lines[i]) > 30 and lines[i].strip()[0:29] == '| Time step number          :':
                             structure += 1
                             start_frame_structure_info.append(i)
                             scf_converged = False
                   elif len(lines[i]) > 44 and lines[i].strip()[0:44] == 'Structure to be used in the next time step:': 
                             if (len(energy) - 1) < structure:
                                       comments.append([])
                                       lattice.append([])
                                       atoms.append([])
                                       energy.append(99.999999)
                                       charge.append([0.0,False])
                                       found_energy = False
                   elif lines[i].strip() == 'Self-consistency cycle converged.':
                             if scf_converged == False:
                                       scf_converged = True
                   elif lines[i].strip()=='Total atomic forces (unitary forces cleaned) [eV/Ang]:':
                             if scf_converged == True:
                                       start_forces.append(i)
                             elif scf_converged == False:
                                       start_forces.append('not converged')
                                       exit()
                   elif len(lines[i]) > 33 and lines[i].strip()[0:33] == '| Total energy                  :':
                             if energy[structure] == 99.999999 and scf_converged:
                                       energy[structure]=float(lines[i].strip().split()[6])*eV2Eh
                                       found_energy=True

    # First time step is read twice: once in input and once in MD traj and thus it has to be removed one time
    if performing_MD and len(start_frame_structure_info):
               start_frame_structure_info.pop(1)

    # Check if everything is correct
    if not performing_relaxation and not performing_MD:
            print('ERROR: Neither a relaxation nor a MD run. Use "aims_out" read function.')
            exit()
    if found_energy==False:
        print('No energy found!!!')
        exit()
    if start_frame_structure_info==None:
        print('Frame not found!!!')
        exit()
    
    # Read general data like charge, if spin polarized 
    for i in np.arange(start_control_info,start_control_info+22): #to get all important informations, charge, functional, k grid etc
         if len(lines[i].split())>2:
                   if lines[i].split()[0].strip()=='Charge':
                             for structure in range(len(charge)):
                                      charge[structure]=[float(lines[i].split()[2].strip()[:6]),True]

    # Read data for each structure of the MD traj or the relaxation steps
    for structure in range(len(start_frame_structure_info)):
         if not no_cell_needed:
                    got_lattice=False
         # Read structure data (atomic positions and lattice)
         for i in np.arange(start_frame_structure_info[structure],len(lines)):
                   if len(lines[i].split())>2:
                             # Find the initial structure
                             if structure == 0:
                                       if lines[i].split()[2].strip()=='Species':
                                                 line=lines[i].strip().split()
                                                 atoms[structure].append(atom(element=str(line[3]),x=float(line[4])*Ang2Bohr,y=float(line[5])*Ang2Bohr,z=float(line[6])*Ang2Bohr))
#                                                 print(lines[i])
                             elif structure > 0:
                                       if lines[i].split()[0].strip()=='atom':
                                                 line=lines[i].strip().split()
                                                 atoms[structure].append(atom(element=str(line[4]),x=float(line[1])*Ang2Bohr,y=float(line[2])*Ang2Bohr,z=float(line[3])*Ang2Bohr))
#                                                 print(lines[i])
                     
                             if got_lattice==False:
#                                       print(lines[i].strip()=='| No unit cell requested.', structure == 0, structure > 0, lines[i].strip())
                                       if structure == 0 and lines[i].split()[1].strip()=='Unit':
                                                           got_lattice=True
                                                           line1=lines[i+1].strip().split()
                                                           line2=lines[i+2].strip().split()
                                                           line3=lines[i+3].strip().split()
                                                           lattice[structure]=np.array([[float(line1[1])*Ang2Bohr,float(line1[2])*Ang2Bohr,float(line1[3])*Ang2Bohr],[float(line2[1])*Ang2Bohr,float(line2[2])*Ang2Bohr,float(line2[3])*Ang2Bohr],[float(line3[1])*Ang2Bohr,float(line3[2])*Ang2Bohr,float(line3[3])*Ang2Bohr]])
                                       elif structure > 0 and lines[i].split()[0].strip()=='lattice_vector':
                                                           got_lattice=True
                                                           line1=lines[i].strip().split()
                                                           line2=lines[i+1].strip().split()
                                                           line3=lines[i+2].strip().split()
                                                           lattice[structure]=np.array([[float(line1[1])*Ang2Bohr,float(line1[2])*Ang2Bohr,float(line1[3])*Ang2Bohr],[float(line2[1])*Ang2Bohr,float(line2[2])*Ang2Bohr,float(line2[3])*Ang2Bohr],[float(line3[1])*Ang2Bohr,float(line3[2])*Ang2Bohr,float(line3[3])*Ang2Bohr]])
                                       elif lines[i].strip()=='| No unit cell requested.':
                                                 got_lattice=True
                                                 no_cell_needed = True
                   if len(atoms[structure]) >= n_atoms:
                             if got_lattice==False:
                                       print('WARNING: No lattice found for frame {:4d}, but seems to need one. Lattice is not printed in MD time steps. Used input lattice.'.format(structure))
                                       comments[structure].append('WARNING: No lattice found for frame {:4d}, but seems to need one. Lattice is not printed in MD time steps. Used input lattice.'.format(structure))
                                       for i in np.arange(start_structure_info,len(lines)):
                                                 if len(lines[i].split()) > 2 and lines[i].split()[1].strip()=='Unit':
                                                           line1=lines[i+1].strip().split()
                                                           line2=lines[i+2].strip().split()
                                                           line3=lines[i+3].strip().split()
                                                           lattice[structure]=np.array([[float(line1[1])*Ang2Bohr,float(line1[2])*Ang2Bohr,float(line1[3])*Ang2Bohr],[float(line2[1])*Ang2Bohr,float(line2[2])*Ang2Bohr,float(line2[3])*Ang2Bohr],[float(line3[1])*Ang2Bohr,float(line3[2])*Ang2Bohr,float(line3[3])*Ang2Bohr]])
                                                           break
                             break

# Read forces
         if start_forces != None:
            for i in np.arange(start_forces[structure],start_forces[structure]+n_atoms+2):
                if len(lines[i].split())>1:
                    if lines[i].split()[0].strip()=='|':
                        count+=1
                        line=lines[i].strip().split()
                        atoms[structure][count-1].force=[float(line[2])*f_conversion,float(line[3])*f_conversion,float(line[4])*f_conversion]
                        atoms[structure][count-1].default_f=False
         count=0

    print(len(comments), len(energy), len(charge), len(lattice), len(atoms))

    return(comments,energy,charge,lattice,atoms)

####################################################################################################

def read_lammps_input(infile,options):
    '''
    function to read in lammps input files used by lammps. if infile is none, the default file name 
    is struct.geo.
    '''
    if infile == None:
        infile = 'struct.geo'
    atoms=[[]]
    comments=[[]]
    energy=[99.999999]
    charge=[[0.0,False]]
    got_lattice=0
    lattice=np.zeros([3,3])
    start_atoms=None
    elements={}
    inv_element_masses = {v: k for k, v in element_masses.items()}
    start_masses=None
    elements_from_options=get_options(options, 'element_order')
    with open(infile,'r') as f:
        lines=f.readlines()
    comments=[[lines[0].strip()]]
    for i in np.arange(1,len(lines)):  #get everything but atoms
        line=lines[i].strip().split()
        if len(line) ==1:
            if line[0]=='Atoms':
                start_atoms=i
            elif line[0]=='Masses':
                start_masses=i
        elif len(line) ==2:
            if line[1]=='atoms':
                n_atoms=line[0]
        elif len(line) ==4:
            if line[3]=='xhi':
                lattice[0,0]=(float(line[1])-float(line[0]))/Bohr2Ang
                got_lattice+=1
            elif line[3]=='yhi':
                lattice[1,1]=(float(line[1])-float(line[0]))/Bohr2Ang
                got_lattice+=1
            elif line[3]=='zhi':
                lattice[2,2]=(float(line[1])-float(line[0]))/Bohr2Ang
                got_lattice+=1
        elif len(line)==6:
            if line[5]=='yz':
                lattice[1,0]=float(line[0])/Bohr2Ang
                lattice[2,0]=float(line[1])/Bohr2Ang
                lattice[2,1]=float(line[2])/Bohr2Ang
                got_lattice+=1
    if elements_from_options==None:
        print('guessing elements from atomic masses! Specify atomic order via option: element_order=XX,XX,XX if this fails!')
        for i in np.arange(start_masses,start_atoms): #get masses
            line=lines[i].strip().split()
            if len(line) ==2:
                elements.update({int(line[0]):inv_element_masses[float(line[1])] if float(line[1]) in inv_element_masses else inv_element_masses[min(inv_element_masses.keys(), key=lambda k: abs(k-float(line[1])))]})
        print('guessed elements: {}'.format(elements))
    else:
        ele_count=1
        for i in elements_from_options.strip().split(','):
            elements.update({ele_count:str(i)})
            ele_count+=1
    for i in np.arange(start_atoms,len(lines)):
        line=lines[i].strip().split()
        if len(line) ==5:
            atoms[0].append(atom(element=elements[int(line[1])],x=float(line[2])/Bohr2Ang,y=float(line[3])/Bohr2Ang,z=float(line[4])/Bohr2Ang))
    return(comments,energy,charge,[lattice],atoms)

####################################################################################################

def write_input_data(comments,energy,charge,lattice,atoms,outfile,frame,options):
    '''
    Writes structures in RuNNer format. If outfile is None, defaults to input.data.
    '''
    if outfile == None:
        outfile='input.data'
    lines=[]
    for i in frame:
        comments[i]=check_for_default(i,energy[i],charge[i],atoms[i],comments[i],options)
        lines.append('begin')
        for j in np.arange(0,len(comments[i])):
            lines.append('comment '+comments[i][j])
        if len(lattice[i]) > 1:
            for j in np.arange(0,len(lattice[i])):
                lines.append('lattice {:12.6f} {:12.6f} {:12.6f}'.format(lattice[i][j][0],lattice[i][j][1],lattice[i][j][2]))
        for j in np.arange(0,len(atoms[i])):
            lines.append('atom {:12.6f} {:12.6f} {:12.6f}  {:2} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}'.format(atoms[i][j].xyz[0],atoms[i][j].xyz[1],atoms[i][j].xyz[2],atoms[i][j].element,atoms[i][j].atom_charge,atoms[i][j].atom_energy,atoms[i][j].force[0],atoms[i][j].force[1],atoms[i][j].force[2]))
        lines.append('energy  '+str(energy[i]))
        lines.append('charge  '+str(charge[i][0]))
        lines.append('end')
    f=open(outfile,'w')
    for i in lines:
        f.write(i+'\n')
    f.close

####################################################################################################

def write_geometry_in(comments,energy,charge,lattice,atoms,outfile,frame,options):
    '''
    Writes structures in fhi-aims geometry.in format. If outfile is None, defaults to geometry.in. 
    Only writes one frame, first if none is specified.
    '''
    if outfile == None:
        outfile='geometry.in'
    if len(frame)>1:
        print("WARNING: Trajectory was found, will write structures consecutively to files with the following output format: <output_string>.<structure_number>")
    for i in frame:
        lines=[]
        for j in np.arange(0,len(comments[i])):
            lines.append('#'+comments[i][j])
        if len(lattice[i]) > 1:
            for j in np.arange(0,len(lattice[i])):
                lines.append('lattice_vector {:12.6f} {:12.6f} {:12.6f}'.format(lattice[i][j][0]*Bohr2Ang,lattice[i][j][1]*Bohr2Ang,lattice[i][j][2]*Bohr2Ang))
        for j in np.arange(0,len(atoms[i])):
            lines.append('atom {:12.6f} {:12.6f} {:12.6f}  {:2}'.format(atoms[i][j].xyz[0]*Bohr2Ang,atoms[i][j].xyz[1]*Bohr2Ang,atoms[i][j].xyz[2]*Bohr2Ang,atoms[i][j].element))
        if len(frame) == 1:
            f=open(outfile, mode='w')
        else:
            f=open(outfile+"."+str(i+1), mode='w')
        for j in lines:
            f.write(j+'\n')
        f.close

####################################################################################################

def write_lammps_input(comments,energy,charge,lattice,atoms,outfile,frame,options):
    '''
    Writes structures in lammps input format. If outfile is None, defaults to struc.geo. Elements 
    will get numbers in order of  tomic number if not specified by option: element_order=XX,XX,XX
    '''
    if outfile == None:
        outfile='struc.geo'
    if len(frame)>1:
        print("WARNING: Trajectory was found, will write structures consecutively to files with the following output format: <output_string>.<structure_number>")
    for i in frame:    
        lines=[]
        elements=[]
        for j in atoms[i]:
            if j.element not in elements:
                elements.append(j.element)
        mass=[]
        for j in elements:
            mass.append(element_masses[j])
        elements=[elements for mass, elements in sorted(zip(mass,elements))]
        elements_from_options=get_options(options, 'element_order')
        if elements_from_options==None:   
            element_numbers=np.arange(1,len(elements)+1)
        else:
            number=0
            elements=[]
            element_numbers=[]
            for element in elements_from_options.strip().split(','):
                number+=1
                elements.append(element)
                element_numbers.append(number)
        element_dictionary={}
        for j in np.arange(0,len(elements)):
            element_dictionary.update({elements[j]:element_numbers[j]})
        lines.append('Generated by RuNNerUC, original comment lines: ')
        for j in np.arange(0,len(comments[i])):
            lines[0]=lines[0]+str(comments[i][j])
        lines.append('')
        lines.append('{} atoms'.format(len(atoms[i])))
        lines.append('{} atom types'.format(len(elements)))
        lines.append('')
        if len(lattice[i]) > 1:
            zcx_a=(lattice[i][0][0]**2+lattice[i][0][1]**2+lattice[i][0][2]**2)**0.5   
            zcx_b=(lattice[i][1][0]**2+lattice[i][1][1]**2+lattice[i][1][2]**2)**0.5 
            zcx_c=(lattice[i][2][0]**2+lattice[i][2][1]**2+lattice[i][2][2]**2)**0.5 

            zcx_vectora=np.array([lattice[i][0][0],lattice[i][0][1],lattice[i][0][2]])
            zcx_vectorb=np.array([lattice[i][1][0],lattice[i][1][1],lattice[i][1][2]])
            zcx_vectorc=np.array([lattice[i][2][0],lattice[i][2][1],lattice[i][2][2]])

            zcx_alpha=np.arccos(np.dot(zcx_vectorb,zcx_vectorc)/(np.linalg.norm(zcx_vectorb) * np.linalg.norm(zcx_vectorc)))
            zcx_beta=np.arccos(np.dot(zcx_vectora,zcx_vectorc)/(np.linalg.norm(zcx_vectora) * np.linalg.norm(zcx_vectorc)))
            zcx_gamma=np.arccos(np.dot(zcx_vectora,zcx_vectorb)/(np.linalg.norm(zcx_vectora) * np.linalg.norm(zcx_vectorb)))

            zcx_lx=zcx_a
            zcx_xy=zcx_b*np.cos(zcx_gamma)
            zcx_xz=zcx_c*np.cos(zcx_beta)
            zcx_ly=(zcx_b**2-zcx_xy**2)**0.5
            zcx_yz=(zcx_b*zcx_c*np.cos(zcx_alpha)-zcx_xy*zcx_xz)/zcx_ly
            zcx_lz=(zcx_c**2-zcx_xz**2-zcx_yz**2)**0.5

            lines.append('0.0 {:12.6f} xlo xhi'.format(zcx_lx*Bohr2Ang))
            lines.append('0.0 {:12.6f} ylo yhi'.format(zcx_ly*Bohr2Ang))
            lines.append('0.0 {:12.6f} zlo zhi'.format(zcx_lz*Bohr2Ang))
            lines.append('{:12.6f} {:12.6f} {:12.6f} xy xz yz'.format(zcx_xy*Bohr2Ang,zcx_xz*Bohr2Ang,zcx_yz*Bohr2Ang))

        lines.append('')
        lines.append('Masses')
        lines.append('')
        for j in np.arange(0,len(elements)):
            try:
                element_masses[elements[j]]
            except:
                print('Element in command line options not available. Maybe it is a typo, the unkown element was: {}'.format(elements[j]))
                exit()
            lines.append('{} {}'.format(element_numbers[j],element_masses[elements[j]]))
        lines.append('')
        lines.append('Atoms')
        lines.append('')
        for j in np.arange(0,len(atoms[i])):
            try:
                element_dictionary[atoms[i][j].element]
            except:
                print('Element in structure file that was not specified in command line options. Either all or no atoms have to be specified. Missing atoms were: {}'.format(atoms[i][j].element))
                exit()
            lines.append('{:5} {:3} {:12.6f} {:12.6f} {:12.6f}'.format(j+1,element_dictionary[atoms[i][j].element],atoms[i][j].xyz[0]*Bohr2Ang,atoms[i][j].xyz[1]*Bohr2Ang,atoms[i][j].xyz[2]*Bohr2Ang))
        if len(frame) == 1:
            f=open(outfile, mode='w')
        else:
            f=open(outfile+"."+str(i+1), mode='w')
        for i in lines:
            f.write(i+'\n')
        f.close

####################################################################################################

def write_lammpstrj(comments, energy, charge, lattice, atoms, outfile, frame, options):
  '''
  Function to write lammpstrj file. The length unit is Angstrom. The forces fx, fy, and fz are in
  eV/Angstrom. The charge q and atomic energy E are not converted by default. The units of q, E, and
  fx, fy, and fz can be changed in the beginning of the function. If outfile is None, the default
  file name is output.lammpstrj. Lattice vectors must be given in order to use the output format.
  '''
  q_conversion = 1
  E_conversion = 1
  f_conversion = Eh2eV/Bohr2Ang

  if outfile == None:
    outfile = 'output.lammpstrj'

  with open(outfile, 'w') as f:
    for i in frame:
      try:
        check = lattice[i][2][2]
        comments[i]=check_for_default(i, energy[i], charge[i], atoms[i], comments[i],options)
        n_atoms = len(atoms[i])
        lattice[i] = np.array(lattice[i])
        a = np.linalg.norm(lattice[i][0])
        b = np.linalg.norm(lattice[i][1])
        c = np.linalg.norm(lattice[i][2])
        cos_alpha = np.dot(lattice[i][1], lattice[i][2])/b/c
        cos_beta = np.dot(lattice[i][0], lattice[i][2])/a/c
        cos_gamma = np.dot(lattice[i][0], lattice[i][1])/a/b
        xy = b*cos_gamma
        lx = a
        xz = c*cos_beta
        ly = np.sqrt(b**2-xy**2)
        yz = (b*c*cos_alpha-xy*xz)/ly
        lz = np.sqrt(c**2-xz**2-yz**2)
        x_lo_bound = np.array([0.0, xy, xz, xy+xz]).min()
        x_hi_bound = lx+np.array([0.0, xy, xz, xy+xz]).max()
        y_lo_bound = np.array([0.0, yz]).min()
        y_hi_bound = ly+np.array([0.0, yz]).max()
        z_lo_bound = 0.0
        z_hi_bound = lz
        for j in np.arange(len(comments[i])):
          print(comments[i][j])
        f.write('ITEM: TIMESTEP\n{0}\nITEM: NUMBER OF ATOMS\n{1}\nITEM: BOX BOUNDS xy xz yz pp pp pp\n{2:12.6f} {3:12.6f} {4:12.6f}\n{5:12.6f} {6:12.6f} {7:12.6f}\n{8:12.6f} {9:12.6f} {10:12.6f}\nITEM: ATOMS id element x y z q E fx fy fz\n'.format(i, n_atoms, x_lo_bound*Bohr2Ang, x_hi_bound*Bohr2Ang, xy*Bohr2Ang, y_lo_bound*Bohr2Ang, y_hi_bound*Bohr2Ang, xz*Bohr2Ang, z_lo_bound*Bohr2Ang, z_hi_bound*Bohr2Ang, yz*Bohr2Ang))
        for j in range(n_atoms):
          f.write('{0:6} {1:2} {2:12.6f} {3:12.6f} {4:12.6f} {5:12.6f} {6:12.6f} {7:12.6f} {8:12.6f} {9:12.6f}\n'.format(j+1, atoms[i][j].element, atoms[i][j].xyz[0]*Bohr2Ang, atoms[i][j].xyz[1]*Bohr2Ang, atoms[i][j].xyz[2]*Bohr2Ang, atoms[i][j].atom_charge*q_conversion, atoms[i][j].atom_energy*E_conversion, atoms[i][j].force[0]*f_conversion, atoms[i][j].force[1]*f_conversion, atoms[i][j].force[2]*f_conversion))
      except IndexError:
        print('ERROR: Lattice vectors are not given for structure {0} (first structure equals structure 0)'.format(i))
        exit()

####################################################################################################

def sort_by_atomic_number(elements):
    '''
    Function to sort an array of element strings by atomic number. 
    Uses the element-masses global dict for ordering (notice that dicts are internally unordered so we need an intermediate here).
    Ex:
    ("O", "Zn", "H", "O") -> ("H", "O", "O", "Zn")
    '''
    order=[]
    for el in elements:
        if el in elements_sorted: #check that the element string is correct
            order.append(elements_sorted.index(el))
        else:
            print("WARNING: Element string is wrong, was {}, exiting".format(el))
            sys.exit(2)

    #sort original elements according to order array
    my_sorted_elements=[el for o, el in sorted(zip(order, elements))]

    return my_sorted_elements

####################################################################################################

#Skeleton for other VASP formats
def read_vasp_poscar(infile,options):
    '''
    Need to implement changing from fractional to cartesian coordinates, skipping element line, and the scaling factor if it's not 1.0
    '''
    elements_from_options=get_options(options, 'element_order')

    myin=open(infile, mode="r")

    #Defaults
    energy=[99.999999]
    charge=[[0.0,False]]

    #comment line
    comments=[[myin.readline().strip()]]

    #scaling_factor
    sf=float(myin.readline().strip())

    #lattices
    lattice=[None,]
    l=[]
    for i in range(0, 3):
        line=myin.readline().strip()
        spline=line.split()
        l.append([sf*float(spline[j])/Bohr2Ang for j in range(0, 3)]) #notice that we need to multiply by the scaling factor
    lattice[-1]=np.array(l)

    #check if next line is element line or count line
    line=myin.readline().strip()
    spline=line.split()
    if spline[0].isdigit():   #there is no element line
        has_element_line=False
    elif spline[0].isalpha(): #there is an element line (redundant on purpose)
        has_element_line=True
    else: #error out
        print("ERROR: Format of POSCAR file is not correct, exiting")
        sys.exit(2)

    #element line
    if has_element_line:
        unique_elements=[el for el in spline if el[0]!="!"] #drop comments if there are some
        #count line, read next line
        count=myin.readline().strip().split()
        count=[int(c) for c in count if c[0]!="!"]
    elif elements_from_options is not None: #check if it was given in the options
        order_elements=elements_from_options.strip().split(',')
        unique_elements=set(order_elements)
        if len(order_elements)!=len(unique_elements): #if the lengths don't match, the set removed some duplicates
            print("WARNING: Repeated elements in element_order, exiting")
            sys.exit(2)
        else:
            unique_elements=order_elements #need to unset since set destroys order

        #count_line, was the splitted line we just read
        count=[int(c) for c in spline if c[0]!="!"]
    else:
        print("ERROR: No elements defined for POSCAR, either in file itself or from element_order option, exiting")
        sys.exit(2)

    #Cartesian or Direct
    line=myin.readline().strip()
    if line[0] in ["S", "s"]: #Selective Dynamics
        print("WARNING: Selective Dynamics activated in POSCAR file")
        #toggle to next line:
        line=myin.readline().strip()

    if line[0] in ["C", "c", "K", "k"]: #Cartesian
        is_cartesian=True
    else: #Direct/Fractional
        print("WARNING: Switching to fractional coordinates for POSCAR reading, check the results carefully!")
        is_cartesian=False

    #finally, atoms
    atoms=[[],]
    #Get element: same for both procedures, we can do it after retrieving the positions for simplicity's sake
    #for c in count:
    #   for i in range(0, c):
    #element is the index of c in unique_elements
    natoms=sum(count)
    for i in range(0, natoms):
        spline=myin.readline().strip().split()
        pos=[sp for sp in spline[0:3]]
        if is_cartesian: #no need to convert units
            #need sf
            pos=[sf*float(p)/Bohr2Ang for p in pos]
        else:
            #sf is already included in the lattice vectors
            #need to convert from fractional to direct coords using the lattice vectors
            frac_pos=[float(p) for p in pos]
            pos=[frac_pos[ax]*np.array(lattice[-1][ax]) for ax in range(0, 3)]
            pos=pos[0]+pos[1]+pos[2]
            pos=[p for p in pos] #don't need to correct Bohr2Ang here, since it has already been done for the lattice itself
        myatom=atom(element="XX", x=pos[0], y=pos[1], z=pos[2])
        atoms[-1].append(myatom)

    running_tally=0
    for enumc, c in enumerate(count):
        current_element=unique_elements[enumc]
        for i in range(0, c):
            atoms[-1][running_tally].element=current_element
            running_tally+=1
    
    if running_tally!=natoms:
        print("ERROR: assigned elements and number of atoms don't match in POSCAR")
        sys.exit(2)    
    
    myin.close()
    return(comments, energy, charge, lattice, atoms)

####################################################################################################

def read_vasp_xdatcar():
    '''
    VASP format for trajectory simulations. But one has to convert the direct to cartesian coordinates!
    '''
    pass

####################################################################################################

def read_vasp_outcar(infile, options):
    '''
    Function to read OUTCAR files (Simple and trajectories) from the Vienna ab initio simulation
    package (VASP), which is the convention. Nevertheless, other given names will be handled.
    Energies are in eV, Forces in eV/Ang, coordinates in Ang.
    This format is a read-only function!!
    '''
    if infile == None:
        infile = "OUTCAR"

    comments             = [[]]
    lattice              = []
    energy_free          = 0.0
    atoms                = [[]]
    energy               = [99.999999]
    charge               = [[0.0,False]]

    has_vasp_entry       = False
    got_lattice          = False
    got_atom_charge      = False

    atom_charge          = [[]]
    atom_energy          = 0.0 # No atom energies in OUTCAR file!

    structure            = 0
    line_number          = 0

    number_of_atoms      = 0
    atoms_per_element    = []
    atomtypes            = []
    positions_and_forces = [[]]

    found_optimization = False

    input_file=open(infile,'r')

    input_lines=input_file.readlines()

    for line in input_lines:

        # comments + initialization
        if line.startswith(' vasp'):
            structure = 1
            has_vasp_entry = True
            comments.append([])
            comments[structure].append(line.split()[0]) #version
            lattice.append([])
            atoms.append([])
            energy.append(99.999999)
            charge.append([0.0,False])
            atom_charge.append([])
            positions_and_forces.append([])

        if 'IBRION' in line: # check if its SPC, MD or optimization
            vasp_mode = int(line.split()[2])
            if vasp_mode == -1 or vasp_mode == 0:
                found_optimization = False
            elif vasp_mode == 1 or vasp_mode == 2 or vasp_mode == 3 or vasp_mode == 44:
                found_optimization = True
            else:
                print("IBRION setting {} cannot be handled by this converter, only available for -1 (Single Point Calculations), 0 (Molecular Dynamics), 1 (Quasi-Newton Relaxation), 2 (Conjugate Gradient Relaxation), 3 (Steepest Descent Relaxation) and 44 (Transition State optimization)".format(vasp_mode))
                sys.exit(2)

        if '  GGA  ' in line: # functional
            comments[structure].append(line.strip())

        if 'ENCUT' in line: # energy cutoff
            comments[structure].append(line.strip())

        if 'IVDW' in line: # vdW correction
            comments[structure].append(line.strip())

        if 'VDW_SR' in line: # vdW parameter
            comments[structure].append(line.strip())

        if 'ISPIN' in line: # spin polarized calculation?
            comments[structure].append(line.strip())

        if 'ISMEAR' in line: # electronic smearing method
            comments[structure].append(line.strip())

        if 'PREC' in line: # algorithm precision
            comments[structure].append(line.strip())

        if 'LASPH' in line: # aspherical Exc?
            comments[structure].append(line.strip())

        if 'NKPTS' in line: # number of K-Points
            comments[structure].append(line.strip())

        if 'VRHFIN' in line: # VRHFIN
            comments[structure].append(line.strip())

        if '   TITEL' in line: # element symbols + number of elements; spaces due to VASP 5.3.5
            comments[structure].append(line.strip())
            atomtypes.append(line.split()[3].split('_')[0].split('.')[0])

        if 'NIONS ' in line: # number of atoms
            number_of_atoms = int(line.split()[-1])

        if 'ions per type' in line: # atoms per element
            atoms_per_element_dummy=[]
            for i in range(len(atomtypes)):
                atoms_per_element_dummy.append([int(x) for x in line.split()[4+i:5+i]])
                atoms_per_element.extend(atoms_per_element_dummy[i])

        if 'direct lattice vectors' in line: # lattice vectors
            if got_lattice==False:
                got_lattice=True
                lattice_all=[]
                for i in range(3):
                    lattice_all.append([float(x) * Ang2Bohr for x in input_lines[line_number + 1 + i].split()[0:3]])
                lattice.append(np.array(lattice_all[:]))

        if 'POSITION' in line: # cartesian coordinates and forces
            for i in range(number_of_atoms):
                positions_and_forces[structure].append([[float(x) * Ang2Bohr for x in input_lines[line_number + 2 + i].split()[0:3]], 
                                                [float(x) * eV2Eh * Bohr2Ang for x in input_lines[line_number + 2 + i].split()[3:6]]])

        if line.lower().startswith('  free  energy   toten'): # E w/ vdW
            energy_free=float(line.split()[-2]) * eV2Eh
            energy[structure]=energy_free

        if 'Hirshfeld charges' in line: # atomic charges and charge of molecule
            #print(line)
            got_atom_charge=True
            atom_charge_dummy=[]
            for i in range(number_of_atoms):
                atom_charge_dummy.append([float(x) for x in input_lines[line_number + 3 + i].split()[2:3]])
                atom_charge[structure].extend(atom_charge_dummy[i])

            charge[structure][0] = sum(atom_charge[structure])
            charge[structure][1] = True

        if line.startswith('  energy  without entropy'): # maybe change to keyword "Iteration" to avoid additional entries for all arrays except lattice!
            structure += 1
            comments.append([])
            #energy_free=0.0
            atoms.append([])
            energy.append(99.999999)
            charge.append([0.0,False])
            got_lattice = False
            atom_charge.append([])
            positions_and_forces.append([])

        line_number += 1

    if has_vasp_entry == False:
        print("Error: 'vasp' expression was not found in file, check if input file is valid!")

    symbol=[] # create array with element symbols
    for element in range(len(atomtypes)): 
        for number in range(int(atoms_per_element[element])):
            symbol.append(atomtypes[element])

    if charge[structure][1] != True: # create array with atomic charges
        for s in range(structure-1):
            for n in range(number_of_atoms):
                atom_charge[s+1].append(0.0)

    for i in range(structure-1): # get proper atoms array to pass
        for j in range(number_of_atoms):
            atoms[i+1].append(atom(element=str(symbol[j]),\
                x=positions_and_forces[i+1][j][0][0],\
                y=positions_and_forces[i+1][j][0][1],\
                z=positions_and_forces[i+1][j][0][2],\
                fx=positions_and_forces[i+1][j][1][0],\
                fy=positions_and_forces[i+1][j][1][1],\
                fz=positions_and_forces[i+1][j][1][2],\
                atom_energy=float(atom_energy),\
                atom_charge=atom_charge[i+1][j]))

    if found_optimization == True:
        comments[1].append("VASP optimization file found, all optimization steps are included")
        print("WARNING: VASP optimization file found, all optimization steps are included; this message was also added as a comment to the output file")

    input_file.close()

    return(comments[1:-1],energy[1:-1],charge[1:-1],lattice[1:],atoms[1:-1]) # return not the first entry because of former convention and skip last entry due to not optimal assignment of arrays (number of structures problematic)!

####################################################################################################

def write_vasp_poscar(comments, energy, charge, lattice, atoms, outfile, frame, options):
    '''
    The POSCAR format for VASP is rather finicky, and will bring trouble if there are any extra blank lines or such. Comments can be added with !:

    FCC CuZn free comment line
    1.0 !scaling factor, 1.0 for just Angstrom units
    0.5 0.5 0.0
    0.0 0.5 0.5
    0.5 0.0 0.5
    Cu Zn !this line is optional, and can be present or absent
    1 1 !amount of each element, as given by POTCAR. We'll set the default to atomic weight order, but choosable with an option
    Cartesian !only the C matters here, can also be Direct for fractional coords
    0.0 0.0 0.0 !these can sometimes have more columns, defining frozen degrees of freedom as T/F
    0.1 0.1 0.1

    Information: comments, lattices, elements, positions
    Not available: forces, charges, energies --> No need to check for defaults
    Because of the way the elements correspond to atoms (the first X atoms correspond to the first number, and so on), we need to be a bit careful when encoding/decoding the element arrays.

    The format is a bit harder to read since the element line is optional, and the coordinates can also be given in fractional coordinates (Direct instead of Cartesian) which you have to then convert back into cartesian using the lattice vectors and some lin. alg. (see VASP manual).

    POSCAR should only ever contain one structure (other files such as XDATCAR can contain trajectories), so this should be checked when reading/writing.
    This file can also contain velocities but we don't care about those.
    CONTCAR is the exact same format as POSCAR so this can be used for both.

    Options:
    element_order=H,Na,S,O,N #define aribtrary element correspondence to POTCAR, default is atomic weight
    '''
    
    #Check if more than one frame has been asked for, POSCAR should in principle only contain one frame
    if len(frame)>1:
        print("WARNING: Trajectory was found, will write structures consecutively to files with the following output format: <output_string>.<structure_number>")

    elements_from_options=get_options(options, 'element_order')
    if elements_from_options is None: #not provided, sort by default atomic weight
        sort_default=True
    else:
        sort_default=False

    #myout=open(outfile, mode='w')

    for iframe in frame:
        #check for non-periodic structures, if found skip
        if None in lattice[iframe]:
            print("Non-periodic structure", iframe, "was skipped")
            continue

        if len(frame) == 1:
            myout=open(outfile, mode='w')
        else:
            myout=open(outfile+"."+str(iframe+1), mode='w')
            

        #comment line
        comb_comment=" ".join(comments[iframe])
        myout.write(comb_comment+"\n")
 
        #scaling factor
        myout.write("1.0\n") #always 1.0 for Angstrom scales
 
        #lattices
        for l in lattice[iframe]:
            myout.write("{:12.6f} {:12.6f} {:12.6f}\n".format(l[0]*Bohr2Ang, l[1]*Bohr2Ang, l[2]*Bohr2Ang))
 
        #element line
        present_elements=[a.element for a in atoms[iframe]]
        if sort_default:
            print("Sorting atoms by atomic weight for VASP")
            unique_elements=set(present_elements) #Get only unique elements: [H, H, O, O, O] -> [H, O]
            unique_elements=sort_by_atomic_number(unique_elements)
        else:
            order_elements=elements_from_options.strip().split(',')
            unique_elements=set(order_elements)
            if len(order_elements)!=len(unique_elements): #if the lengths don't match, the set removed some duplicates
                print("WARNING: Repeated elements in element_order, exiting")
                sys.exit(2)
            else:
                unique_elements=order_elements #need to unset since set destroys order
 
        myout.write(" ".join("{:3s}".format(ue) for ue in unique_elements)+"\n")
        
        #count line
        counts=[]
        for el in unique_elements:
            c=present_elements.count(el)
            counts.append(str(c))
        myout.write(" ".join("{:3s}".format(c) for c in counts)+"\n")
        
        #Cartesian or Direct
        myout.write("Cartesian\n")

        #Coordinates finally
        for el in unique_elements: #for each element
            for at in atoms[iframe]: #check for every atom
                if at.element==el: #if the atom belongs to that element
                    pos=at.xyz
                    myout.write("{:12.6f} {:12.6f} {:12.6f}\n".format(pos[0]*Bohr2Ang, pos[1]*Bohr2Ang, pos[2]*Bohr2Ang)) #output it
        myout.close()

####################################################################################################

def read_xyz(infile, options):
    '''
    Reading in of an xyz file format, the first line of the xyz file contains number of atoms, second is a blank line or could be a comment line. The third line
    onwards, the geometry of a structure is given in cartesian coordinates preceded by the atom name. 
    '''
    if infile==None:
        infile='input.xyz'
    atoms=[[]]
    comments=[[]]
    energy=[99.999999]
    charge=[[0.0,False]]
    lattice=[[]]
    with open(infile,'r') as f:
        lines=f.readlines()
    number_atoms=[]
    next_structure=0
    for i in np.arange(0,len(lines)):
        if next_structure==i:
            if len(lines[i].strip().split()) != 1:
                print('Format of .xyz file is wrong')
                exit()
            number_atoms.append(i)
            next_structure=next_structure+int(lines[i].strip())+2
    number_atoms.append(len(lines))
    for i in np.arange(0,len(number_atoms)-1):
        comments.append([])
        lattice.append([])
        energy.append([])
        energy[i]=99.999999
        charge.append([])
        charge[i]=[0.0,False]
        comments[i].append(lines[int(number_atoms[i])+1].strip())
        if len(atoms)<i+1:
            atoms.append([])
        for j in np.arange(int(number_atoms[i])+2,int(number_atoms[i+1])):
            line=lines[j].strip().split()
            atoms[i].append(atom(element=str(line[0]),x=float(line[1])/Bohr2Ang,y=float(line[2])/Bohr2Ang,z=float(line[3])/Bohr2Ang))



    return(comments[:-1],energy[:-1],charge[:-1],lattice[:-1],atoms)

####################################################################################################
def write_xyz(comments,energy,charge,lattice,atoms,outfile,frame,options):
    '''
    Writes files in the xyz format. If outfile is None, defaults to input.xyz. 
    '''
    if outfile == None:
        outfile='input.xyz'
    lines=[]
    for i in frame:
        lines.append(str(len(atoms[i])))
        lines.append(''.join(comments[i]))
        for j in np.arange(0,len(atoms[i])):
            lines.append('{:2} {:12.6f} {:12.6f} {:12.6f}'.format(atoms[i][j].element,atoms[i][j].xyz[0]*Bohr2Ang,atoms[i][j].xyz[1]*Bohr2Ang,atoms[i][j].xyz[2]*Bohr2Ang))
    f=open(outfile,"w")
    for i in lines:
        f.write(i+'\n')
    f.close()



####################################################################################################

def define_frame(frame,number_of_frames):
    '''
    Returns an array for write_functions to use. The array is a list of frames to be used, indexed from 0 to tot_num_frames-1.
    This indexing is done thru keywords or a simple number. Assumes frames are indexed from 1.
    Examples: File with 100 frames, numbered 1-100
        All:        [0, 1, 2, 3, ..., 98, 99]
        First:      [0]
        Last:       [99]
        Every5:     [0, 5, 10, ..., 90, 95] #Notice it doesn't include the last frame even tho it's a multiple, but includes the first one
        Every6:     [0, 6, 12, ..., 90, 96]
        50:         [49]
        Random2:    [12, 87] or any two random values between 0 and 99
        Slice20-25: [19, 20, 21, 22, 23, 24]
    If you want to add an extra mode, you can do it here.
    '''
    if frame == 'All':
        frame=list(np.arange(0,number_of_frames))
    elif frame == 'Last':
        frame=[number_of_frames-1]
    elif frame == 'First':
        frame=[0]
    elif frame.startswith('Random'):
        ran=int(frame[6:]) #number of random structures to get
        if ran>number_of_frames:
            print("ERROR: Asked for more Random frames than are available, exiting")
            sys.exit(2)
        from random import sample
        frame=sample(list(range(0, number_of_frames)), ran)
        frame=sorted(frame)
    elif frame[0:5] == 'Every':
        try:
            frame=list(np.arange(0,number_of_frames,int(frame[5:])))
        except:
            print('EveryX was used without X being a number: Every2 Every4 or Every10')
            sys.exit(2)
    elif frame.startswith("Slice"): #Slice10-20 -> nds are inclusive, so you would get 10, 11, ..., 19, 20 for a total of 11 structures
        ranges=frame[5:]
        ranges=ranges.split("-")
        ranges=[int(r) for r in ranges]
        if (ranges[0])<=0 or (ranges[1]<=0):
            print("ERROR: Frame slice selection wrong, can't use 0 or negative frames, exiting")
            sys.exit(2)
        if ranges[1]>number_of_frames:
            print("WARNING: Set endof frame slice to be larger than the total number of frames. Setting it to the number of frames and continuing")
            ranges[1]=number_of_frames
        frame=list(range(ranges[0]-1, ranges[1]+1-1)) #-1 to convert to zero indexing, +1 to make the end point inclusive
    else:
        try:
            frame=[int(frame)-1]
            if frame[0] <0:
                print('0 or negative frame numbers cant be used. First frame in structure is frame 1')
                sys.exit(2)
        except: 
            print('Frame ist neither a number nor First or Last or All or RandomX or SliceX-Y or EveryX (you will get every Xth frame)')
            sys.exit(2)
        if frame[0] >= len(energy):
            print('Frame is too large')
            sys.exit(2)

    #print(frame)
    return(frame)

####################################################################################################

def check_for_default(structure,energy,charge,atoms,comments,options):
    if get_options(options, 'silent')=='True':
        return(comments)
    set_default=[]
    if abs(energy-99.999999)<thres:
        set_default.append('energy')
        print('WARNING!!! Energy of structure {} was set to default value 99.999999'.format(structure+1))
    if charge[1]==False:
        set_default.append('charge')
        print('WARNING!!! Charge of structure {} was set to default value 0.0'.format(structure+1)) 
    for i in atoms:
        try:
            if i.default_f==True:
                set_default.append('force of atom')
                print('WARNING!!! Force of atom in structure {} was set to default value 99.999999'.format(structure+1)) 
                break
        except:
            continue
    if len(set_default)>=1:
        already_in_comments=False
        for i in comments:
            if 'were set to default values' in i:
                already_in_comments=True
        if already_in_comments==False:
            comments.append('Values for {} were set to default values'.format(set_default))
    return(comments)
    
####################################################################################################
########################################## Script ##################################################
####################################################################################################

### Dictionaries with available formats for reading and writing files

dict_read= {'runner':read_input_data,  'lammpstrj':read_lammpstrj,  'aims_geo':read_geometry_in, 'lammps_input':read_lammps_input,  
            'vasp_poscar':read_vasp_poscar,'vasp_outcar':read_vasp_outcar,'aims_out':read_aims_output,'xyz':read_xyz,'aims_traj':read_aims_traj
            }
dict_write={'runner':write_input_data, 'lammpstrj':write_lammpstrj, 'aims_geo':write_geometry_in,'lammps_input':write_lammps_input, 
            'vasp_poscar':write_vasp_poscar,'xyz':write_xyz
            }

if __name__=='__main__':
    ### check if inline comments were given
    if len(sys.argv)>=3:
        format_read=sys.argv[1]
        format_write=sys.argv[2]
    elif len(sys.argv)>=2:
        if sys.argv[1]=='-l':
            print('available reading formats:')
            for i in dict_read:
                print(i, dict_read[i].__doc__)
            print('avilable writing formats:')
            for i in dict_write:
                print(i, dict_write[i].__doc__)
            exit()
        else:
            print(help1)
            sys.exit(2)
    else:
        print(help1)
        sys.exit(2)
    
    ### check if format for reading is available and set it
    if format_read in dict_read:
        read_func=dict_read[format_read]
    else:
        print('Reading format not found')
        print('available reading formats:')
        for i in dict_read:
            print(i, dict_read[i].__doc__)
        print('avilable writing formats:')
        for i in dict_write:
            print(i, dict_write[i].__doc__)
        sys.exit()
    
    ### check if format for writing is available and set it
    if format_write in dict_write:
        write_func=dict_write[format_write]
    else:
        print('Writing format not found')
        print('available writing formats:')
        for i in dict_write:
            print(i, dict_write[i].__doc__)
        print('available reading formats:')
        for i in dict_read:
            print(i, dict_read[i].__doc__)
        sys.exit()
    
    ### get the rest of the inline comments
    infile,outfile,frame,given_comments,options=main(sys.argv[3:])
    
    ### use the reading function
    comments,energy,charge,lattice,atoms=read_func(infile,options)
    
    ### add inline comments, appends command line comments to the original comments
    ### optional: discard option discards all options except the new ones given in command line
    discard_comments=get_options(options,'discard_comments')
    if discard_comments==None:
        if given_comments!=None:
            for i in np.arange(len(comments)):
                comments[i].append(given_comments)
    elif discard_comments=='All':
        if given_comments!=None:
            for i in np.arange(len(comments)):
                comments[i]=[given_comments]
        else:
            for i in np.arange(len(comments)):
                comments[i]=[]
    else:
        print('discard_comments option only works with with this input: discard_comments=All')
        exit() 
    
    ### check if the length of every output is the same, every structure has everything
    if not len(comments) == len(energy) == len(charge) == len(lattice) == len(atoms):
        print('Energy, Comments, Charge, Lattice, Atoms arrays have different lengths')
        sys.exit()
    defined_frame=define_frame(frame,len(energy))
    
    ### use the writing function
    write_func(comments,energy,charge,lattice,atoms,outfile,defined_frame,options)
