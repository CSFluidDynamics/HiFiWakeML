from torch.utils.data import Dataset
import numpy as np
import os

def cropnpad(dat, x_start, y_start):
    ndat = np.pad(dat, pad_width=((0,0),(50,50),(100,50)),mode='edge')
    return ndat[:,x_start:x_start+280,y_start:y_start+450]

class FlorisLesDataset(Dataset):

    def __init__(self,les_dir, floris_dir, augment=True):
        self.fpath = floris_dir
        self.lpath = les_dir
        self.files = os.listdir(self.lpath)
        self.augment= augment
        print(self.files)
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        #print(fn[-6:-4])
        
        c = np.zeros(5)
        try:
            ws = int(fn[-6:-4])
            c[0] = ws
            if fn[0] == 'W':
                c[1] = 1.0
            elif fn[0] == 'N':
                c[2] = 1.0
            elif fn[6] == 'W':
                c[3] = 1.0
            else:
                c[4] = 1.0
        except:
            None
        if self.augment:
            x_start = np.random.randint(0,100)
            y_start = np.random.randint(0,150)
            return {'x':cropnpad(np.load(os.path.join(self.fpath,self.files[idx])),x_start,y_start),
                    'y':cropnpad(np.load(os.path.join(self.lpath,self.files[idx])),x_start,y_start),
                    'fn':self.files[idx], 'c':c}
        else:
            return {'x':np.load(os.path.join(self.fpath,self.files[idx])),
                    'y':np.load(os.path.join(self.lpath,self.files[idx])),
                    'fn':self.files[idx], 'c':c}

class FlorisLesParamDataset(Dataset):
    def __init__(self, wdmap_dir, freeflow_dir, floris_dir, les_dir, augment=True):
        self.wdmap_dir = wdmap_dir
        self.freeflow_dir = freeflow_dir
        self.floris_dir = floris_dir
        self.les_dir = les_dir
        self.files = os.listdir(self.floris_dir)
        self.augment = augment
        print(self.files)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        fn = self.files[idx]
        ws = int(fn[-6:-4])
        wd = fn[:-7]
        freeflow = np.load(os.path.join(self.freeflow_dir, '%d.npy'%ws))
        wdmap = np.load(os.path.join(self.wdmap_dir, '%s.npy'%wd))
        floris = np.load(os.path.join(self.floris_dir,fn))/ws
        try:
            les = np.load(os.path.join(self.les_dir,fn))
            les_map = 1
        except:
            les = np.zeros_like(freeflow)
            les_map = 0
        if self.augment:
            x_start = np.random.randint(0, 100)
            y_start = np.random.randint(0, 150)
            freeflow = cropnpad(freeflow, x_start, y_start)
            wdmap = cropnpad(wdmap, x_start, y_start)
            floris = cropnpad(floris, x_start, y_start)
            if les_map==1:
                les = cropnpad(les, x_start, y_start)
        return {
            'freeflow': freeflow,
            'wdmap': wdmap,
            'floris': floris,
            'les': les,
            'les_map': les_map,
            'ws': ws,
            'fn':self.files[idx]
        }

class MaskedLesParamDataset(Dataset):
    def __init__(self, wdmap_dir, freeflow_dir, mask_dir, les_dir):
        self.wdmap_dir = wdmap_dir
        self.freeflow_dir = freeflow_dir
        self.mask_dir = mask_dir
        self.les_dir = les_dir
        self.files = os.listdir(self.les_dir)
        print(self.files)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        fn = self.files[idx]
        ws = int(fn[-6:-4])
        wd = fn[:-7]
        freeflow = np.load(os.path.join(self.freeflow_dir, '%d.npy'%ws))
        wdmap = np.load(os.path.join(self.wdmap_dir, '%s.npy'%wd))
        
        les = np.load(os.path.join(self.les_dir,fn))
        if les.shape[0] == 284:
            les = les[142:]
        try:
            mask = np.load(os.path.join(self.mask_dir,fn[:-4]+'-pseudo-label-mask.npy'))
        except:
            mask = np.ones_like(les)
            print(fn + ' no mask')
        
        
        return {
            'freeflow': freeflow,
            'wdmap': wdmap,
            'les': les,
            'mask': mask,
            'ws': ws,
            'fn':self.files[idx]
        }
