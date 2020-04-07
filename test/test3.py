import pylab as pl
import numpy as np


gaussian = lambda x: np.exp(-(0.5-x)**2/1.5)
x = np.arange(-2,2,0.01)
y = gaussian(x)

pl.figure()
pl.plot(x,y)
pl.xlabel('x value')
pl.ylabel('y value')
pl.title('Gaussian')
pl.show()
