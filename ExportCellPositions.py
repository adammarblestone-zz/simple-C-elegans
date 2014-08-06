# Adam Marblestone using code from OpenWorm

'''Draws the connectome in 3D.'''

import neuroml
import neuroml.loaders as loaders
import neuroml.writers as writers

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import csv

outfile = "OpenWorm/cell_coords.csv"

connectome_spreadsheet = "OpenWorm/connectome.csv"

all_the_cell_names = ["ADAL","ADAR","ADEL","ADER","ADFL","ADFR","ADLL","ADLR","AFDL","AFDR","AIAL","AIAR","AIBL","AIBR","AIML","AIMR","AINL","AINR","AIYL","AIYR","AIZL","AIZR","ALA","ALML","ALMR","ALNL","ALNR","AQR","AS1","AS10","AS11","AS2","AS3","AS4","AS5","AS6","AS7","AS8","AS9","ASEL","ASER","ASGL","ASGR","ASHL","ASHR","ASIL","ASIR","ASJL","ASJR","ASKL","ASKR","AUAL","AUAR","AVAL","AVAR","AVBL","AVBR","AVDL","AVDR","AVEL","AVER","AVFL","AVFR","AVG","AVHL","AVHR","AVJL","AVJR","AVKL","AVKR","AVL","AVM","AWAL","AWAR","AWBL","AWBR","AWCL","AWCR","BAGL","BAGR","BDUL","BDUR","CEPDL","CEPDR","CEPVL","CEPVR","DA1","DA2","DA3","DA4","DA5","DA6","DA7","DA8","DA9","DB1","DB2","DB3","DB4","DB5","DB6","DB7","DD1","DD2","DD3","DD4","DD5","DD6","DVA","DVB","DVC","FLPL","FLPR","HSNL","HSNR","I1L","I1R","I2L","I2R","I3","I4","I5","I6","IL1DL","IL1DR","IL1L","IL1R","IL1VL","IL1VR","IL2DL","IL2DR","IL2L","IL2R","IL2VL","IL2VR","LUAL","LUAR","M1","M2L","M2R","M3L","M3R","M4","M5","MCL","MCR","MI","NSML","NSMR","OLLL","OLLR","OLQDL","OLQDR","OLQVL","OLQVR","PDA","PDB","PDEL","PDER","PHAL","PHAR","PHBL","PHBR","PHCL","PHCR","PLML","PLMR","PLNL","PLNR","PQR","PVCL","PVCR","PVDL","PVDR","PVM","PVNL","PVNR","PVPL","PVPR","PVQL","PVQR","PVR","PVT","PVWL","PVWR","RIAL","RIAR","RIBL","RIBR","RICL","RICR","RID","RIFL","RIFR","RIGL","RIGR","RIH","RIML","RIMR","RIPL","RIPR","RIR","RIS","RIVL","RIVR","RMDDL","RMDDR","RMDL","RMDR","RMDVL","RMDVR","RMED","RMEL","RMER","RMEV","RMFL","RMFR","RMGL","RMGR","RMHL","RMHR","SAADL","SAADR","SAAVL","SAAVR","SABD","SABVL","SABVR","SDQL","SDQR","SIADL","SIADR","SIAVL","SIAVR","SIBDL","SIBDR","SIBVL","SIBVR","SMBDL","SMBDR","SMBVL","SMBVR","SMDDL","SMDDR","SMDVL","SMDVR","URADL","URADR","URAVL","URAVR","URBL","URBR","URXL","URXR","URYDL","URYDR","URYVL","URYVR","VA1","VA10","VA11","VA12","VA2","VA3","VA4","VA5","VA6","VA7","VA8","VA9","VB1","VB10","VB11","VB2","VB3","VB4","VB5","VB6","VB7","VB8","VB9","VB10","VB11", "VC1", "VC2", "VC3","VC4","VC5", "VD1", "VD2", "VD3", "VD4", "VD5", "VD6", "VD7", "VD8", "VD9", "VD10", "VD11", "VD12", "VD13"]
print "Number of neurons:"
print(len(all_the_cell_names))

cells_connections = {} # synaptic
cells_gaps = {} # gap junctional

connection_nums = {}
gap_nums = {}

GABA_expressing = []
Cholinergic = []
Glutamatergic = []

def main():      
        
    coords = [] # list of 3D coordinates of all the neurons
        
    # get each cell's soma coords
    for cellName in all_the_cell_names:
        inFile = "OpenWorm/straightenedNeuroML2/" + cellName + ".nml"

        # open the cell
        doc = loaders.NeuroMLLoader.load(inFile)
        if not doc.cells:
            sys.exit(1)
        cell = doc.cells[0]

        # find the soma coordinates
        c = get3DPosition(cell, 0, 0.5)
        # print cellName + "\t" + str(c[0]) + "\t" + str(c[1]) + "\t" + str(c[2])
        coords.append(c)
        cells_connections[cellName] = [c]
        cells_gaps[cellName] = [c]
        
    # write the coordinates to a file
    outfile = open("OpenWorm/nominal_cell_coords.txt", "w")
    for name in all_the_cell_names:
        outfile.write("%s\t%f\t%f\t%f\n" % (name, cells_connections[name][0][0], cells_connections[name][0][1], cells_connections[name][0][2]))
    outfile.close()
        
    # read connectivity data
    readSynapticConnectome()
    readGapConnectome()
    
    # set up the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim3d(-50,50)
    ax.set_ylim3d(-400,400)
    ax.set_zlim3d(-30,30)
    plt.xlabel("x")
    plt.ylabel("y")
    
    # plot the soma coordinates
    ax.plot([c[0] for c in coords], [c[1] for c in coords], [c[2] for c in coords], 'o')

    # plot the synaptic connections
    k = cells_connections.keys()
    for key in k:
        s = cells_connections[key][0]    
        end_coords_list = [cells_connections[p][0] for p in cells_connections[key][1:]]
        for i in range(len(end_coords_list)):
            e = end_coords_list[i]
            p = cells_connections[key][1:][i]
            num = float(connection_nums[key][p])
            if key in GABA_expressing:
                ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color='c', linestyle='dashed', linewidth = num/5)
            elif key in Cholinergic or key in Glutamatergic:
                ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color='g', linestyle='dashed', linewidth = num/10)
            else:                
                ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color='m', linestyle='dashed', linewidth = num/10)
            
            
    # plot the gap junctions
    k = cells_gaps.keys()
    for key in k:
        s = cells_gaps[key][0]
        end_coords_list = [cells_gaps[p][0] for p in cells_gaps[key][1:]]
        for i in range(len(end_coords_list)):
            e = end_coords_list[i]
            p = cells_gaps[key][1:][i]
            num = float(gap_nums[key][p])
            ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color='k', linestyle='dotted', linewidth = num/5)
    
    plt.show()
    
def readGapConnectome():
    
    for k1 in all_the_cell_names:
        gap_nums[k1] = {}
        for k2 in all_the_cell_names:
            gap_nums[k1][k2] = 0
    
    with open(connectome_spreadsheet, 'rU') as f:
        r = csv.reader(f)
        for row in r:
            if row[2] != "Send":
                cells_gaps[row[0]].append(row[1])
                gap_nums[row[0]][row[1]] = row[3]
    
def readSynapticConnectome():
    
    for k1 in all_the_cell_names:
        connection_nums[k1] = {}
        for k2 in all_the_cell_names:
            connection_nums[k1][k2] = 0
    
    with open(connectome_spreadsheet, 'rU') as f:
        r = csv.reader(f)
        for row in r:
            if row[2] == "Send":
                cells_connections[row[0]].append(row[1])
                connection_nums[row[0]][row[1]] = row[3]
                if "GABA" in row[4] and not row[0] in GABA_expressing:
                    GABA_expressing.append(row[0])
                if "Acetylcholine" in row[4] and not row[0] in Cholinergic:
                    Cholinergic.append(row[0])
                if "Glutamate"  in row[4] and not row[0] in Glutamatergic:
                    Glutamatergic.append(row[0])

def getSegmentIds(cell):
    seg_ids = []
    for segment in cell.morphology.segments:
        seg_ids.append(segment.id)

    return seg_ids

def get3DPosition(cell, segment_index, fraction_along):
    seg = cell.morphology.segments[segment_index]
 
    end = seg.distal

    start = seg.proximal
    if start is None:
        segs = getSegmentIds(cell)
        seg_index_parent = segs.index(seg.parent.segments)
        start = cell.morphology.segments[seg_index_parent].distal

    fx = fract(start.x, end.x, fraction_along)
    fy = fract(start.y, end.y, fraction_along)
    fz = fract(start.z, end.z, fraction_along)

    #print "(%f, %f, %f) is %f between (%f, %f, %f) and (%f, %f, %f)"%(fx,fy,fz,fraction_along,start.x,start.y,start.z,end.x,end.y,end.z)

    return fx, fy, fz

def fract(a, b, f):
    return a+(b-a)*f
    
if __name__ == '__main__':
    main()
