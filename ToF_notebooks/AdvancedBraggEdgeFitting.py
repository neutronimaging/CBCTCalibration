import numpy as np
from numpy import pi, r_, math, random
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.special import erfc
from lmfit import Model
from numpy import loadtxt

def term0(t,a2,a6):
    return  a2 * (t - a6)

def term1(t,a2,a5,a6):
    return ((a5 - a2) / 2) * (t - a6)

def term3(t,t0,sigma):
    return erfc(-((t-t0)/(sigma * math.sqrt(2))))

def term4(t,t0,alpha,sigma):
    return np.exp(-((t-t0)/alpha) + ((sigma*sigma)/(2*alpha*alpha)))

def term5(t,t0,alpha,sigma):
    return erfc(-((t-t0)/(sigma * math.sqrt(2))) + sigma/alpha)

#TO DO: add the option to fitt either shape (?) or not
def BraggEdgeShape2(t,t0,alpha,sigma,a1,a2,a5,a6):
    return a1 + term0(t,a2,a6) + term1(t,a2,a5,a6) * (1-(term3(t,t0,sigma) - term4(t,t0,alpha,sigma)* term5(t,t0,alpha,sigma)))

def BraggEdgeShape(t,t0,alpha,sigma,a1,a2,a5,a6):
    return a1 + term0(t,a2,a6) + term1(t,a2,a5,a6) * ((term3(t,t0,sigma) - term4(t,t0,alpha,sigma)* term5(t,t0,alpha,sigma)))

#I am not sure that all these steps are necessary, most likely one can also only define one and decide how many to fit: TODO understand this 
def AdvancedBraggEdgeFittingFirstStep(t,a1,a6):
    return a1 + term0(t,a2_f,a6) + term1(t,a2_f,a5_f,a6) * (1-(term3(t,t0_f,sigma_f) - term4(t,t0_f,alpha_f,sigma_f)* term5(t,t0_f,alpha_f,sigma_f)))

def AdvancedBraggEdgeFittingSecondStep(t,a2,a5):
    return a1_f + term0(t,a2,a6_f) + term1(t,a2,a5,a6_f) * (1-(term3(t,t0_f,sigma_f) - term4(t,t0_f,alpha_f,sigma_f)* term5(t,t0_f,alpha_f,sigma_f)))

def AdvancedBraggEdegFittingThirdStep(t,t0,sigma,alpha):
    return a1_f + term0(t,a2_f,a6_f) + term1(t,a2_f,a5_f,a6_f) * (1-(term3(t,t0,sigma) - term4(t,t0,alpha,sigma)* term5(t,t0,alpha,sigma)))

def AdvancedBraggEdegFittingFourthStep(t,a1,a2,a5,a6):
    return a1 + term0(t,a2,a6) + term1(t,a2,a5,a6) * (1-(term3(t,t0_f,sigma_f) - term4(t,t0_f,alpha_f,sigma_f)* term5(t,t0_f,alpha_f,sigma_f)))

def AdvancedBraggEdegFittingFifthStep(t,t0,sigma,alpha): #this is just the same as the third
    return a1_f + term0(t,a2_f,a6_f) + term1(t,a2_f,a5_f,a6_f) * (1-(term3(t,t0,sigma) - term4(t,t0,alpha,sigma)* term5(t,t0,alpha,sigma)))

def AdvancedBraggEdegFittingAll(t,t0,sigma,alpha,a1,a2,a5,a6):
    return a1 + term0(t,a2,a6) + term1(t,a2,a5,a6) * (1-(term3(t,t0,sigma) - term4(t,t0,alpha,sigma)* term5(t,t0,alpha,sigma)))

def AdvancedBraggEdgeFitting(myspectrum, myrange, est_pos, est_sigma, est_alpha, bool_print):
    
    #get the part of the spectrum that I want to fit
#    mybragg = -1*np.log(myspectrum[myrange[0]:myrange[1]]/np.max(myspectrum[myrange[0]:myrange[1]]))
#    mybragg = mybragg/np.max(mybragg)# iniziamo senza rumore aggiunto
    mybragg= myspectrum[myrange[0]:myrange[1]]
    plt.figure
    plt.plot(mybragg)
    
    #first step: estimate the linear function before and after the Bragg Edge
    
    est_pos=est_pos-myrange[0]
    t=np.linspace(0,np.size(mybragg)-1,np.size(mybragg))
    t_before= t[0:est_pos-int(est_pos*0.1)]
    bragg_before=mybragg[0:est_pos-int(est_pos*0.1)]
    t_after= t[est_pos+int(est_pos*0.1):-1]
    bragg_after=mybragg[est_pos+int(est_pos*0.1):-1]
    
    [slope_before, interception_before] = np.polyfit(t_before, bragg_before, 1)
    [slope_after, interception_after] = np.polyfit(t_after, bragg_after, 1)
    
#    plt.figure
#    plt.plot(t_before,bragg_before,'.k')
#    plt.plot(t_after,bragg_after,'.k')
#    plt.plot(t,mybragg)

#    plt.plot(t,interception_before+slope_before*t)
#   plt.plot(t,interception_after+slope_after*t)
    
    #first guess of paramters
    t0_f=est_pos
    a2_f=slope_before
    a5_f=slope_after
    a6_f=t0_f-(2*mybragg[est_pos+int(est_pos*0.25)]/(a5_f-a2_f)) # I assume edge intensity is equal to 1 as I am normalizing THAT IS NOT ALWAYS THE CASE 
    a1_f=interception_before+a2_f*a6_f
#     sigma_f=-1
#     alpha_f=-10

    sigma_f = est_sigma
    alpha_f = est_alpha
    # method='trust_exact'
    # method='nelder' #not bad
    # method='differential_evolution' # needs bounds
    # method='basinhopping' # not bad
    # method='lmsquare' # this should implement the Levemberq-Marquardt but it says Nelder-Mead method (which should be Amoeba)
    method ='least_squares' # default and it implements the Levenberg-Marquardt
    

    gmodel = Model(AdvancedBraggEdegFittingAll)
    
    params = gmodel.make_params(t0=t0_f,sigma=sigma_f, alpha=alpha_f, a1=a1_f, a2=a2_f, a5=a5_f, a6=a6_f)
    params['alpha'].vary = False
    params['sigma'].vary = False
    params['t0'].vary = False
    params['a2'].vary= False
    params['a5'].vary = False
    
    
    result1 = gmodel.fit(mybragg, params, t=t, method=method, nan_policy='propagate')
#    print(result1.fit_report())
    
#    plt.figure
#    plt.plot(t, mybragg, 'b-')
#    plt.plot(t, result1.init_fit, 'k--')
#    plt.plot(t, result1.best_fit, 'r-')
#    plt.show()
    
    a1_f=result1.best_values.get('a1')
    a6_f=result1.best_values.get('a6')
    
    params = gmodel.make_params(t0=t0_f,sigma=sigma_f, alpha=alpha_f, a1=a1_f, a2=a2_f, a5=a5_f, a6=a6_f)
    params['alpha'].vary = False
    params['sigma'].vary = False
    params['t0'].vary = False
    params['a1'].vary= False
    params['a6'].vary = False
        

    result2 = gmodel.fit(mybragg, params, t=t, method=method, nan_policy='propagate')
#    print(result2.fit_report())
#    plt.figure
#    plt.plot(t, mybragg, 'b-')
#    plt.plot(t, result2.init_fit, 'k--')
#    plt.plot(t, result2.best_fit, 'r-')
#    plt.show()

    a2_f = result2.best_values.get('a2')
    a5_f = result2.best_values.get('a5')
    
    params = gmodel.make_params(t0=t0_f,sigma=sigma_f, alpha=alpha_f, a1=a1_f, a2=a2_f, a5=a5_f, a6=a6_f)
    params['a2'].vary = False
    params['a5'].vary = False
    params['a1'].vary= False
    params['a6'].vary = False
    params['sigma'].vary= False
    params['alpha'].vary = False


    result3=gmodel.fit(mybragg, params, t=t, method=method, nan_policy='propagate')
#    print(result3.fit_report())
#    plt.figure
#    plt.plot(t, mybragg, 'b-')
#    plt.plot(t, result3.init_fit, 'k--')
#    plt.plot(t, result3.best_fit, 'r-')
#    plt.show()
    
    t0_f=result3.best_values.get('t0')
#     sigma_f=result3.best_values.get('sigma')
#     alpha_f=result3.best_values.get('alpha')

    params = gmodel.make_params(t0=t0_f,sigma=sigma_f, alpha=alpha_f, a1=a1_f, a2=a2_f, a5=a5_f, a6=a6_f)
    params['a2'].vary = False
    params['a5'].vary = False
    params['a1'].vary= False
    params['a6'].vary = False
#     params['t0'].vary= False

    result4=gmodel.fit(mybragg, params, t=t, nan_policy='propagate',method=method)
                       


#    print(result4.fit_report())
#    plt.figure
#    plt.plot(t, mybragg, 'b-')
#    plt.plot(t, result4.init_fit, 'k--')
#    plt.plot(t, result4.best_fit, 'r-')
#    plt.show()
    
    sigma_f=result4.best_values.get('sigma')
    alpha_f=result4.best_values.get('alpha')
    t0_f=result4.best_values.get('t0')
    
    
    params = gmodel.make_params(t0=t0_f,sigma=sigma_f, alpha=alpha_f, a1=a1_f, a2=a2_f, a5=a5_f, a6=a6_f)
    params['t0'].vary = False
    params['sigma'].vary = False
    params['alpha'].vary= False
    
    
    result5 = gmodel.fit(mybragg, params, t=t, nan_policy='propagate', method=method)
#    print(result5.fit_report())
#    plt.figure
#    plt.plot(t, mybragg, 'b-')
#    plt.plot(t, result5.init_fit, 'k--')
#    plt.plot(t, result5.best_fit, 'r-')
#    plt.show()
    
    a1_f =result5.best_values.get('a1')
    a2_f = result5.best_values.get('a2')
    a5_f = result5.best_values.get('a5')
    a6_f = result5.best_values.get('a6')

#     t0_f=result5.best_values.get('t0')
#     sigma_f=result5.best_values.get('sigma')
#     alpha_f=result5.best_values.get('alpha')

    params = gmodel.make_params(t0=t0_f,sigma=sigma_f, alpha=alpha_f, a1=a1_f, a2=a2_f, a5=a5_f, a6=a6_f)
    params['a2'].vary = False
    params['a5'].vary = False
    params['a1'].vary= False
    params['a6'].vary = False
    
    result6= gmodel.fit(mybragg, params, t=t, nan_policy='propagate', method=method)



#     result6=gmodel6.fit(mybragg,t=t, t0=t0_f,sigma=sigma_f, alpha=alpha_f, a1=a1_f, a2=a2_f, a5=a5_f, a6=a6_f, nan_policy='propagate', method=method)
    
#    print(result6.fit_report())
#    plt.figure
#    plt.plot(t, mybragg, 'b-')
#    plt.plot(t, result6.init_fit, 'k--')
#    plt.plot(t, result6.best_fit, 'r-')
#    plt.show()

    t0_f=result6.best_values.get('t0')
    sigma_f=result6.best_values.get('sigma')
    alpha_f=result6.best_values.get('alpha')

    params = gmodel.make_params(t0=t0_f,sigma=sigma_f, alpha=alpha_f, a1=a1_f, a2=a2_f, a5=a5_f, a6=a6_f)

    result7 = gmodel.fit(mybragg, params, t=t, nan_policy='propagate', method=method)
    
    print(result7.fit_report())

    
    t0_f=result7.best_values.get('t0')
    sigma_f=result7.best_values.get('sigma')
    alpha_f=result7.best_values.get('alpha')
    a1_f =result7.best_values.get('a1')
    a2_f = result7.best_values.get('a2')
    a5_f = result7.best_values.get('a5')
    a6_f = result7.best_values.get('a6')
    
    plt.figure
    plt.plot(t, mybragg, 'b-')
    plt.plot(t, result7.init_fit, 'k--')
    plt.plot(t, result7.best_fit, 'r-', linewidth='1.2')
    plt.plot(t, result1.best_fit, '--')
    plt.plot(t, result2.best_fit, '--')
    plt.plot(t, result3.best_fit, '--')
    plt.plot(t, result4.best_fit, '--')
    plt.plot(t, result5.best_fit, '--')
    plt.plot(t, result6.best_fit, '--')
    plt.plot(t[t0_f], result7.best_fit[t0_f],'ok')
    plt.show()
    
    if (bool_print):
        print(result1.fit_report())
        print(result2.fit_report())
        print(result3.fit_report())
        print(result4.fit_report())
        print(result5.fit_report())
        print(result6.fit_report())
        
    
    return {'t0':t0_f, 'sigma':sigma_f, 'alpha':alpha_f, 'a1':a1_f, 'a2':a2_f,'a5':a5_f, 'a6':a6_f, 'final_result':result7}

    