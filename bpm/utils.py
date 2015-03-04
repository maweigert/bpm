
import os
import sys
import time

class StopWatch(object):
    """ stops time in miliseconds

    s = StopWatch()

    s.tic()

    foo()
    
    print t.toc()
    """
    def __init__(self):
        self.times  = dict()
        self._dts  = dict()

    def tic(self,key = ""):
        self._dts[key] = time.time()

    def toc(self,key = ""):
        self.times[key] = 1000.*(time.time()- self._dts[key])
        return self.times[key]

    def __getitem__(self, key,*args):
        return self.times.__getitem__(key,*args)

    def __setitem__(self, key,val):
        self.times[key] = val


    def __repr__(self):
        return "\n".join(["%s:\t%.3f ms"%(str(k),v) for k,v in self.times.iteritems()])

def absPath(myPath):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        return os.path.join(base_path, os.path.basename(myPath))
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, myPath)




if __name__ == '__main__':

    s = StopWatch()

    s.tic()

    time.sleep(1.)
    print "time passed:", s.toc()


    print "absPath: ", absPath(".") 
