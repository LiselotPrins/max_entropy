import numpy as np
from autograd import numpy as anp
from scipy.stats import norm, laplace, expon, pareto, vonmises, rayleigh, cauchy
from scipy.special import iv #modified bessel function of first kind

def pdf_norm(x,m,s2):
    return norm.pdf(x, loc=m, scale=np.sqrt(s2))
def pdf_expon(x,a):
    return expon.pdf(x,scale=1/a)
def pdf_laplace(x,mu,c):
    return laplace.pdf(x, loc=mu, scale=c)
def pdf_pareto(x,a,xm):
    return pareto.pdf(x,a,loc=0,scale=xm)
def pdf_cauchy(x):
    return cauchy.pdf(x)

def pdf_rayleigh(x,loc,a):
    return rayleigh.pdf(x,loc=loc,scale=a)
def pdf_vonmises(x,loc,k):
    return vonmises.pdf(x,loc=loc,kappa=k)



def cdf_norm(x,m,s2):
    return norm.cdf(x, loc=m, scale=np.sqrt(s2))
def cdf_expon(x,a):
    return expon.cdf(x,scale=1/a)
def cdf_laplace(x,mu,c):
    return laplace.cdf(x, loc=mu, scale=c)
def cdf_pareto(x,a,xm):
    return pareto.cdf(x,a,loc=0,scale=xm)
def cdf_cauchy(x):
    return cauchy.cdf(x)

def cdf_rayleigh(x,loc,a):
    return rayleigh.cdf(x,loc=loc,scale=a)
def cdf_vonmises(x,loc,k):
    return vonmises.cdf(x,loc=loc,kappa=k)

### Normal distribution: 1 constraint ###
def f_constraint_normal1(x, mu, s2):
    return np.array([(x-mu)**2])

def b_constraint_normal1(mu, s2):
    return np.array([s2])

def lambda_actual_normal1(mu, s2):
    return np.array([(-0.5/s2)])

### Exponential ###
def f_constraint_expon(x, a):
    return np.array([x])

def b_constraint_expon(a):
    return np.array([1/a])

def lambda_actual_expon(a):
    return np.array([-a])

### Laplace ###
def f_constraint_laplace(x,mu,c):
    return np.array([anp.abs(x-mu)])

def b_constraint_laplace(mu,c):
    return np.array([c])

def lambda_actual_laplace(mu,c):
    return np.array([-1/c])

### Pareto ###
def f_constraint_pareto(x,a,xm):
    return np.array([anp.log(x)])

def b_constraint_pareto(a,xm):
    return np.array([1/a + anp.log(xm)])

def lambda_actual_pareto(a,xm):
    return np.array([-a-1])

### Cauchy ###
def f_constraint_cauchy(x):
    return np.array([anp.log(1+x**2)])

def b_constraint_cauchy():
    return np.array([2*np.log(2)])

def lambda_actual_cauchy():
    return np.array([-1])

### Normal distribution: 2 constraints ###
def f_constraint_normal2(x, mu, s2):
    return np.array([x, x**2])

def b_constraint_normal2(mu, s2):
    return np.array([mu, s2 + mu**2])

def lambda_actual_normal2(mu, s2):
    return np.array([mu/s2, -0.5/s2])

### Rayleigh ###
def f_constraint_rayleigh(x):
    return np.array([x**2, anp.log(x)])

def b_constraint_rayleigh(s2):
    return np.array([2*s2, (np.log(2*s2)- np.euler_gamma)/2])

def lambda_actual_rayleigh(s2):
    return np.array([-0.5/s2,1])

### Von Mises ###
def f_constraint_vonmises(x):
    return np.array([anp.cos(x), anp.sin(x)])

def b_constraint_vonmises(mu, k):
    a = iv(1,k)/iv(0,k)
    return np.array([a * np.cos(mu), a * np.sin(mu)])

def lambda_actual_vonmises(mu, k):
    return np.array([k*anp.cos(mu), k*anp.sin(mu)])

### Skewness-kurtosis ###
def f_constraint_skewkurt(x):
    return np.array([x,x**2,x**3,x**4])

def b_constraint_skewkurt(skew, kurt):
    return np.array([0, 1, skew, kurt])

### Skewness-kurtosis + 6th ###
def f_constraint_skewkurt_6(x):
    return np.array([x,x**2,x**3,x**4,x**8])

def b_constraint_skewkurt_6(skew, kurt,  m10):
    return np.array([0, 1, skew, kurt, m10])