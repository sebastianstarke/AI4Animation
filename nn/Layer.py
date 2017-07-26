import numpy as np
import theano

class Layer(object):
   
    def cost(self, input):
        return 0
   
    def load(self, database, prefix=''):
        for param in self.params:
            param.set_value(database[prefix+param.name].astype(theano.config.floatX), borrow=True)
            
    def save(self, database, prefix=''):
        for param in self.params:
            database[prefix+param.name] = np.array(param.get_value(borrow=True))

        