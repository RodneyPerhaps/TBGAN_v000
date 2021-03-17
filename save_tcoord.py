import numpy as np
import pickle

def main():
    vt = []
    coord_file = '/home/jschen/tcoords'
    with open(coord_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            vt.append([float(line.split(' ')[1]), float(line.split(' ')[2])])
        
    vt = np.array(vt)
    with open('tcoord.pkl', 'wb') as f:
        pickle.dump(vt, f)

if __name__ == '__main__':
    main()
