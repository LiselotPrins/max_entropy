
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
# Autodiff package
from autograd import numpy as anp
from autograd import jacobian, hessian
# Integration function for normalization constant
from scipy.integrate import quad 

class MaxEnt:
    """ A class used to solve a maximum entropy problem and visualize 
    the results.
    
    Attributes
    ----------
    _f_vector : array of functions
        the constraint functions
    _f_param : array of numbers
        extra parameters for functions in _f_vector
    _l, _u : floats
        the lower resp. upper limit of support [l,u]
    _m : int
        the amount of constraints (k in thesis)
    _b_vector : array of floats
        expectations of constraint functions
    _converges : Boolean
        indicator if algorithm has converged to a solution
    _w : array of floats
        weights of gaussian quadrature
    _A : 2D array/matrix
        matrix used in method _Q
    _jacobian, _hessian : functions
        jac/hess of method _Q
    _k_end : int
        amount of iterations for the algorithm until convergence reached
        or until maximum iteration number was reached
    _la : array of floats
        the values of lambda at final iteration; is _la_all[-1,:]
    _la_all : 2D array of floats
        _m columns, _k_end rows
        the i-th column stores lambda values of iteration i of the algo
    _norm_const : float
        value which makes resulting function integrate to 1 on [_l,_u]
        Is calculated in function xxx

    Methods
    ----------
    pdf(x)
        Returns value of ME pdf in value x
    cdf(x)
        Returns value of ME cdf in value x
    visualize_algorithm(title=)
        Plots figure with evolution of lambda_k through algorithm, 
        and print resulting value for lambda.
    visualize_solution(xlim=[l,u], title=, actual_density=None, 
                       actual_param=None, actual_lambda=None):  
        Plots resulting ME distribution. 
        If desired, the 'actual' density can be plotted as well.
        The error of lambda can also be calculated, given the theoretical 
        value.
    calc_error(actual_lambda)
        Returns Euclidian distance between result of algorithm, _la, and
        actual_lambda.

    _algorithm(k_max=100, start=np.zeros(self._m), 
               warning_convergence=True, message_norm=False)
        Runs Algorithm, stores resulting parameters in self._la,
        calculates normalization constant.
    _Q(la_eval)
        Returns value of Q as in Eq. 7 of paper.
    _set_norm_const(message=False)
        Sets normalization constant and, if message is True, prints error 
        of integration.
    
    get_la(), get_convergence()
        Getters
    """

    def __init__(self, support, 
                 n_quadrature, 
                 b_constraints, 
                 f_vector, 
                 f_param=[], 
                 k_max=100, 
                 start=None, 
                 warning_convergence=True, 
                 message_norm=False):
        """Initializes MaxEnt object

        Parameters
        ----------
        support:                The support [l,u] for the distribution. 
                                Array of two numbers.
        n_quadrature:           The number of points in Gaussian quadrature
        b_constraints:          The vector of numbers
        f_vector:               Vector/array of functions,
                                (same shape as b_constraints)
        f_param:                Array of parameters for functions in f_vector
                                (Default: empty)
        k_max:                  Max amount of iterations of the algorithm.
                                If is has not converged by then, it will stop
                                and send warning if warning_convergence is true.
                                (Default: 100)
        start:                  Start vector for algorithm. 
                                (Default: np.zeros([self._m]), which will
                                be set in _algorithm)
        warning_convergence:    If true, and the algorithm doesn't stop 
                                before k_max iterations, a warning is made.
                                (Default: True)
        message_norm:           If true, a message stating the error on 
                                the normalisation constant will be shown.
                                (Default: False)
        """

        # The constraint functions, in an array
        # Notation: [f_1,...,f_m]
        self._f_vector = f_vector 
        self._f_param = f_param

        # The support of the pdf: finite bounded domain
        # Notation: [l,u]
        self._l = support[0]
        self._u = support[1]

        # The amount of constraints
        # Notation: m
        self._m = f_vector(self._l, *f_param).size 

        # The expected values of constraint functions
        # Notation: [b_1,...,b_m], where b_i= E[f_i(X)]
        self._b_vector = b_constraints

        # Indicates if algorithm has converged to a solution
        self._converges = False 

        if len(b_constraints) != self._m:
            raise Exception("The dimension of constraints should match the dimension of the function vector.")


        # Here follows the modified optimization algorithm from Rockinger
        # & Jondeau (2002). The 'Steps' concide with the steps from
        # Algorithm 1 of the paper (p. 126)

        # Step 1:
        # Domain is [l,u], as defined above.
        # z and w are the points resp. weights of the n_quadrature point 
        # gaussian quadrature. Here, z consists of values between 0 and 1.
        z, self._w = np.polynomial.legendre.leggauss(int(n_quadrature))

        # Step 2:
        # Map z (in [0,1]) to x (in [l,u]).
        # Define matrix A with j-th (1<=j<=n) row is (f_1(x_j)-b_1,...,f_m(x_j)-b_m).
        # The function Q is defined a a method of the MaxEnt class.
        # Two member variables are created: the jacobian & Hessian of Q, with
        # the autodifferentiation provided by the autograd package.
        x = ((self._u-self._l)*z + self._u+self._l)/2
        self._A = np.asarray(
            [ self._f_vector(xj,*f_param)-self._b_vector for xj in x])

        self._jacobian = jacobian(self._Q) 
        self._hessian = hessian(self._Q)

        self._algorithm(k_max=k_max, start=start,
                        warning_convergence=warning_convergence, 
                        message_norm=message_norm)

    def pdf(self, x):
        """Returns value of ME pdf in value x
        
        Parameters
        ----------
        x: number or np.array
            values in which to calculate pdf

        Returns
        ----------
        pdf evaluated in x, in the same shape as the input x
        """
        
        # If x is a scalar, calculate pdf in x.
        if isinstance(x, (int, float, np.floating)):
            return np.exp(
                        np.inner(self._la, 
                                self._f_vector(x,*self._f_param)-self._b_vector)
                         )/self._norm_const
        
        # If x is array-like, recursively call pdf on the elements of x.
        # This also works for higher-dimensional arrays.
        if isinstance(x, np.ndarray):
            return np.array([
                self.pdf(a) for a in x
            ])
        
        # No valid argument
        raise TypeError("The argument of 'pdf' should be a scalar or np.array of scalars")
    
    
    def cdf(self, x):
        """Returns cdf of ME in value x

        Parameters
        ----------
        x: number or np.array
            values in which to calculate cdf

        Returns
        ----------
        cdf evaluated in x, in the same shape as the input
        """

        # If x is scalar, calculate cdf in x.
        if isinstance(x, (int, float, np.floating)):
            return quad(lambda a: self.pdf(a), self._l, x)[0]
    
        # If x is array-like, recursively call cdf on the elements of x.
        # This also works for higher-dimensional arrays.
        if isinstance(x, np.ndarray):
            return np.array([
                self.cdf(a) for a in x
            ])

        raise TypeError("The argument of 'cdf' should be a scalar or np.array of scalars")


    def visualize_algorithm(self, title="Evolution of parameters",
                            title_save=None, 
                            figsize=(5,4)):
        """Show a figure which visualizes lambda^i as function of i.

        Parameters
        ----------
        title: str (optional)
            Title for figure
        title_save: str (optional)
            Title for saving figure. By default, the figure is not saved
        figsize : 2d tuple (optional)
            Figure size

        Actions
        ----------
        Make figure showing evolution of Lagrange parameters throughout the 
        algorithm. If title_save is set, this figure is saved.

        Returns
        ----------
        None
        """

        print("The resulting parameters:")
        for i in range(self._m):
            print(f"  Lambda_{i+1} = {self._la[i]:.3f}")

        
        fig, ax1 = plt.subplots(dpi=100, figsize=figsize)
        ax1.set_xlabel('Iteration $i$')
        ax1.set_ylabel('$\\lambda_j^i$')
        ax1.tick_params(axis='y')

        for i in range(self._m):
            ax1.plot(self._la_all[:,i], label = f'{i+1}')

        ax1.legend(loc='upper left', title="$j$")
        ax1.set_title(title)
        ax1.grid(color='w')

        fig.tight_layout()

        if(title_save is not None):
            plt.savefig(title_save)
        plt.show()


    def visualize_solution(self, xlim=None, 
                            title="Maximum entropy density", 
                            title_save=None,
                            figsize=(5,4),
                            actual_density=None, 
                            actual_param=None,
                            actual_lambda=None):
        """Show a figure which visualizes MaxEnt distribution.

        Parameters
        ----------
        xlim : 2d tuple (optional)
            limits for plotting MaxEnt distribution. Default: [l,u]
        title : str (optional)
            Title for figure
        title_save : str (optional)
            Title for saving figure. By default, the figure is not saved
        figsize : 2d tuple (optional)
            Figure size
        actual_density: function (optional)
            Function with arguments x, and parameters actual_param. If specified,
            it is plotted in the figure
        actual_param : tuple or array (optional)
            tuple with parameters of actual_density
        actual_lambda : array with m dimensions (optional)
            if specified, the error is calculated using the euclidian distance
        

        Actions
        ----------
        Make figure showing evolution of Lagrange parameters throughout the 
        algorithm. If title_save is set, this figure is saved. 
        If actual_density is specified, this is also plotted.
        If actual_lambda is specified, the error is printed.

        Returns
        ----------
        None
        """
        if(xlim is None):
            xlim = (self._l, self._u)
        
        N = 500
        xx = np.linspace(*xlim, N)

        yy = self.pdf(xx)

        
        fig, ax1 = plt.subplots(dpi=100, figsize=figsize)
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('pdf')
        ax1.tick_params(axis='y')

        ax1.plot(xx,yy,label="Result algorithm")

        if(actual_density is not None):
            ax1.plot(xx, actual_density(xx, *actual_param), 
                        label="Actual density",linestyle=":")

        ax1.legend(loc='best')
        ax1.set_title(title)
        ax1.grid(color='w')
        ax1.set_xlim(*xlim)
        
        fig.tight_layout()

        if(title_save is not None):
            plt.savefig(title_save)
        plt.show()
        
        if(actual_lambda is not None):
            print(f"The (Euclidian) distance between result and actual parameter:")
            print(f"{self.calc_error(actual_lambda):.3e}")

    def calc_error(self, actual_lambda):
        """Calculates Euclidian distance between resulting parameters and real values"""
        return np.linalg.norm(self._la - actual_lambda)
    
    def _algorithm(self, k_max=100, start=None, 
                   warning_convergence=True, message_norm=False):
        """Executes modified Algo 1 from Rockinger & Jondeau

        If the Algorithm doesn't converge in k_max steps, a message is
        printed. _k_end is then equal to k_max.

        If the Algorithm does converge, _k_end is set to the first k s.t.
        the k-th iteration doesn't change more than 10^-9 w.r.t. k-1-th 
        iteration.

        Sets normalization constant.

        Parameters
        ----------
        k_max: int, optional
            maximum amount of iterations for the algorithm
            (default is 100)
        start: array of floats, optional
            value of la_0, i.e. start value 
            (default is zero vector)
        warning: Boolean, optional

        """

        # This construction is used because 'self' can't be used in 
        # parameter definition of function
        if(start is None):
            start = np.zeros(self._m)

        # Step 3:
        # Initialize values k and la
        la_k = start
        self._la_all = np.array([la_k]) #it will store all la_k's
        k = 1

        self._k_end = k_max # value of k at which algo stops iterating

        #Step 4-7
        while(k < self._k_end):

            # Step 4: calculate jacobian and hessian.
            g = self._jacobian(la_k)
            G = self._hessian(la_k)

            # Step 5: Solve linear system G.d = -g for d
            d = np.linalg.solve(G, -g)

            # Step 6: Update la_k+1 <- la_k +d
            la_k = la_k + d
            self._la_all = np.append(self._la_all, [la_k], axis=0)

            # Step 7: k++
            k = k+1

            # Stop iterating if steps are too small -> convergence reached
            # At least 10 steps, to ensure it doesn't stop too early
            if(k > 10 and np.linalg.norm(d) < 10**(-9)):
                self._k_end = k
                self._converges = True

        # Store results of Lagrange parameterss in _la
        self._la = self._la_all[-1,:]

        if(self._k_end == k_max and warning_convergence):
            print("The algorithm may not be converging.")
            print("Use 'ob._visualize_algorithm()' to inspect.")
        
        self._set_norm_const(message=message_norm)

    def _Q(self, la_eval):
        """Returns value of  Q(la_eval) as in Eq. 7 of paper."""
        return anp.matmul(self._w, anp.exp(anp.dot(self._A, la_eval)))
    
    def _set_norm_const(self, message=False):
        """Set normalization constant and print error of integration"""
        c, s = quad(lambda x: np.exp(np.inner(self._la, 
                                    self._f_vector(x,*self._f_param)-self._b_vector)), 
                    self._l, self._u)
        
        self._norm_const = c

        # error is usually low; below e-7
        if(message):
            print(
                f"Estimate of absolute integration error of norm. const.: {s:.3e}"
                )

    #Getters
    def get_la(self):
        return self._la
    
    def get_converges(self):
        return self._converges