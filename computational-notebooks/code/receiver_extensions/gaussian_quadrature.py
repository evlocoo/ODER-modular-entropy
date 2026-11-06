"""Independent characteristic-function quadrature for six Fock entries."""
import numpy as np
from scipy.integrate import simpson

def rot(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def direct_quadrature(X,Y,N=401,L=7.):
    alpha=0.5*(X[0,0]+X[1,1]+1j*(X[1,0]-X[0,1]))
    beta=0.5*(X[0,0]-X[1,1]+1j*(X[1,0]+X[0,1]))
    V=.5*X@X.T+Y
    x=np.linspace(-L,L,N); xx,yy=np.meshgrid(x,x,indexing='ij'); xi=xx+1j*yy
    # Explicit characteristic function, not the moment contractions in the module.
    n=.5*np.trace(V)-.5; m=.5*(V[0,0]-V[1,1])+1j*V[0,1]
    vacuum=np.exp(-(n+1)*abs(xi)**2+.5*(m.conjugate()*xi**2+m*xi.conjugate()**2))
    z=alpha.conjugate()*xi-beta*xi.conjugate()
    integ=lambda f:simpson(simpson(f,x=x,axis=1),x=x,axis=0)/np.pi
    return np.array([integ(vacuum),integ((1-abs(xi)**2)*vacuum),
        integ((1-abs(z)**2)*vacuum),integ((1-abs(z)**2)*(1-abs(xi)**2)*vacuum),
        integ(z.conjugate()*xi*vacuum),integ(-z.conjugate()*xi.conjugate()*vacuum)])

