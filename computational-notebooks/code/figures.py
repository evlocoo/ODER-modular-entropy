import numpy as np
import matplotlib.pyplot as plt

def modular_bridge(output):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(9.0,3.5),layout='constrained')
    x=np.linspace(0,2.5,500)
    colors=['#174a70','#aa6810','#923b51']
    for b,c in zip([1.,.75,.25],colors):
        A=np.sinh(x)*np.cosh(x)/(b+np.sinh(x)**2)
        axs[0].plot(x,A,color=c,lw=2,label=f'b = {b:g}')
    axs[0].axhline(1,color='0.55',lw=.9,ls=':')
    axs[0].set(xlabel=r'Modular coordinate $\pi s$',ylabel=r'Normalized hazard $A_b$',title='(a) Preparation changes activation',ylim=(0,1.22))
    axs[0].legend(frameon=False,loc='lower right')
    N=np.arange(1,101)
    E=np.sqrt(N+1)*np.tanh(1.)**(N+2)
    g=1/np.cosh(1.)**2
    mask=E<g
    bound=(np.sqrt(2)+2)*E[mask]/(2*(g-E[mask]))
    axs[1].semilogy(N[mask]+1,bound,color='#174a70',lw=2)
    axs[1].axhline(1e-6,color='0.4',ls='--',lw=1,label=r'Tolerance $10^{-6}$')
    n=62;e=np.sqrt(n+1)*np.tanh(1.)**(n+2)
    a=(np.sqrt(2)+2)*e/(2*(g-e))
    axs[1].plot([63],[a],'o',color='#923b51',ms=5)
    axs[1].annotate('63 sites',xy=(63,a),xytext=(72,2e-4),arrowprops=dict(arrowstyle='-',color='0.5'))
    axs[1].set(xlabel='Total sites (source + receiver)',ylabel='Uniform activation-error bound',title=r'(b) Finite receiver: $h=1$, $\Omega T=1$',ylim=(1e-11,1e2))
    axs[1].legend(frameon=False,loc='upper right')
    fig.savefig(output/'modular_preparation_receiver.pdf')
    fig.savefig(output/'modular_preparation_receiver.png',dpi=180)
    return fig
