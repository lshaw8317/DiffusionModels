# -*- coding: utf-8 -*-
"""
https://github.com/JTT94/diffusion_schrodinger_bridge/blob/main/bridge/models/basic/time_embedding.py
https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=8PPsLx4dGCGa
https://github.com/xymm35/ic-summer-project-final-codes/blob/main/final_unconditional_1.ipynb
"""

import torch
from scipy.stats import norm,multivariate_normal,moment
import numpy as np
import torch.nn.functional as F
import tqdm
import pickle
import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
plt.rc('font',**{'serif':['cm']})
plt.style.use('seaborn-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
ms=5
train=True
save=True
normalise=False
likelihood_weighting=False

# Initialize training data
## learning rate
lr=1e-4
eps=0
T=1-eps
M=100
N=int(1e6) #monte carlo error =1e-(3)


def my_round(x,figs):
    power=np.floor(np.log10(np.abs(x))).astype(np.int)
    newx=np.round(x/(10.**(power-figs)),1)
    return newx*10.**(power-figs)

def summary_stats(x,figs=1):
    mean=np.array2string(my_round(np.mean(x.numpy(),axis=0),figs),separator=', ')
    var=np.array2string(my_round(moment(x,moment=2),figs),separator=', ')
    moment3=np.array2string(my_round(moment(x,moment=3),figs),separator=', ')
    moment4=np.array2string(my_round(moment(x,moment=4),figs),separator=', ')
    return (mean,var,moment3,moment4)

def Wasserstein1(u,v):
    '''
    Return W1 distance between two densities u,v
    '''
    return torch.sum(torch.abs(torch.cumsum(u,dim=0)/torch.sum(u)-torch.cumsum(v,dim=0)/torch.sum(v)))/torch.sum(u)

#%%
def get_timestep_embedding(timesteps, embedding_dim=128):
    """
    Taken directly from https://github.com/JTT94/diffusion_schrodinger_bridge/blob/main/bridge/models/basic/time_embedding.py
    """
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)

    emb = timesteps.float() * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, [0,1])

    return emb

class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final = False, activation_fn=F.relu):
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x

class BortoliMLP(torch.nn.Module):

    def __init__(self, mstd, encoder_layers=[32], pos_dim=32, decoder_layers=[256,512,256], x_dim=1):
        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]

        self.net = MLP(2 * t_enc_dim,
                       layer_widths=decoder_layers +[x_dim],
                       activate_final = False,
                       activation_fn=torch.nn.LeakyReLU())

        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())

        self.x_encoder = MLP(x_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())
        self.marginal_prob_std=mstd
    def forward(self, x, t):
        if len(x.shape)==1:
            x = x.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x)
        h = torch.cat([xemb ,temb], -1)
        out = self.net(h) 
        return out


####Loss function
def loss_function(model, x, marginal_prob_std, mean_map, diff_coeff,eps=eps):
    """The loss function for training score-based generative models.

    Args:
      model: A PyTorch model instance that represents a 
        time-dependent score-based model.
      x: A mini-batch of training data.    
      marginal_prob_std: A function that gives the standard deviation of 
        the perturbation kernel.
      eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand((x.shape[0],1),dtype=torch.float) * (1. - eps) + eps  
    z = torch.randn_like(x) 
    std = marginal_prob_std(random_t)
    m = mean_map(random_t)
    perturbed_x = m*x + z*std
    score = model(perturbed_x, random_t)
    weighting = diff_coeff(perturbed_x,random_t)/std if likelihood_weighting else 1.
    loss = torch.mean(torch.sum((weighting*(score + z))**2, dim=(1)))
    return loss


class Experiment:
    
    def __init__(self,sde,data,dim=1,train=False,normalise=False):
        if sde not in ['Drift', 'VE', 'VP', 'subVP']:
            raise ValueError("sde argument must be one of 'Drift', 'VE', 'VP', 'subVP'")
        data=data+str(dim)+'d'
        if data not in ['Gaussian1d', 'GMM1d', 'Gaussian2d', 'GMM2d','Rosenbrock2d']:
            raise ValueError("data+dim+'d'argument must be one of 'Gaussian1d', 'GMM1d', 'Gaussian2d', 'GMM2d','Rosenbrock2d'")
        self.sde=sde
        self.data=data
        self.dimension=dim
        normalised='Norm' if normalise else ''
        self.tag=sde+data+normalised
        if self.sde=='Drift':
            alpha =  5.
            def marginal_prob_std_fn(t):
                t = torch.tensor(t)
                return torch.sqrt((1-torch.exp(-2*alpha*t))/alpha)
            
            def mean_map_fn(t):
                t = torch.tensor(t)
                return torch.exp(-alpha*t)
            
            def diffcoeff(y,t):
                return np.sqrt(2)
            
            def driftcoeff(y,t):
                return -alpha*y
            y0=torch.randn(size=(N,self.dimension))*np.sqrt(1./alpha)
            title='$dX=\\alpha Xdt+\sqrt{2}dW$'+f', $\\alpha={alpha}$'
        
        ##VE dx=sigma**tdW
        elif self.sde=='VE':
            sigma = 100.
            def marginal_prob_std_fn(t):
                t = torch.tensor(t)
                return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))
            
            def mean_map_fn(t):
                t = torch.tensor(t)
                return torch.ones_like(t)
            
            def diffcoeff(y,t):
                return sigma**(t)
            
            def driftcoeff(y,t):
                return 0
            y0=torch.randn(size=(N,self.dimension))*marginal_prob_std_fn(T)
            title='$dX=\sigma^tdW$'+f', $\sigma={sigma}$'
        
        #VP dx=-(1/2)*(a(1-t/T)+bt/T)dt+sqrt(a(1-t/T)+bt/T)dW
        elif self.sde=='VP':
            a=.1
            b=20.
            helper=lambda t: a*(1.-t/T)+b*t/T
            def marginal_prob_std_fn(t):
                t = torch.tensor(t)
                return torch.sqrt(1.-torch.exp(-helper(t/2.)*t))
            
            def mean_map_fn(t):
                t = torch.tensor(t)
                return torch.exp(-helper(t/2.)*t/2.)
            
            def diffcoeff(y,t):
                return torch.sqrt(helper(t))
            
            def driftcoeff(y,t):
                return -.5*helper(t)*y
            y0=torch.randn(size=(N,self.dimension))
            
            title='$dX=-\\frac{1}{2}(a(1-t/T)+bt/T)Xdt+\sqrt{(a(1-t/T)+bt/T)}dW$'+f', $a={a},b={b}$'
        
        else: # self.sde=='subVP':
            a=.1
            b=20.
            helper=lambda t: a*(1.-t/T)+b*t/T
            def marginal_prob_std_fn(t):
                t = torch.tensor(t)
                return 1.-torch.exp(-helper(t/2.)*t)
            
            def mean_map_fn(t):
                t = torch.tensor(t)
                return torch.exp(-helper(t/2.)*t/2.)
            
            def diffcoeff(y,t):
                return torch.sqrt(helper(t)*(1-torch.exp(-helper(t/2.)*t*2.)))
            
            def driftcoeff(y,t):
                return -.5*helper(t)*y
            y0=torch.randn(size=(N,self.dimension))
           
            title='$dX=-\\frac{1}{2}\\beta(t)Xdt+\sqrt{\\beta(t)(1-\exp(-2\int_0^t\\beta(s)ds))}dW$'+f', $\\beta(t)=(a(1-t/T)+bt/T)$, $a={a},b={b}$'
            
        self.marginal_prob_std_fn=marginal_prob_std_fn
        self.mean_map_fn=mean_map_fn
        self.diffcoeff=diffcoeff
        self.driftcoeff=driftcoeff
        self.title=title
        self.y0=y0
        torch.manual_seed(2022)
        if self.data=='Gaussian1d':    
            mu0=1.0
            sig0=2.0
            true_moments=[mu0,sig0**2,0.,3*sig0**4]
            n_epochs =  50
            batch_size = 500;batches=100;
            
            mlp=BortoliMLP(mstd=self.marginal_prob_std_fn, encoder_layers=[16], pos_dim=16, decoder_layers=[128,128])
            
            mus=torch.tensor([mu0])
            sigs=torch.tensor([sig0])
            x_train=mu0+sig0*torch.randn(batches*batch_size,1)
            datadist = lambda x: torch.as_tensor(norm.pdf(x,loc=mu0,scale=sig0))
            def trueScore(x,t):
                return -(x-mean_map_fn(t)*mu0)/(marginal_prob_std_fn(t)+mean_map_fn(t)**2*sig0**2)
            def trueProb(x,t):
                return torch.as_tensor(norm.pdf(x,loc=mean_map_fn(t)*mu0,scale=torch.sqrt(marginal_prob_std_fn(t)+mean_map_fn(t)**2*sig0**2)))
        
        elif self.data=='GMM1d': #GMM
            n_epochs =  400
            batch_size =  1000;batches=50;
            mlp=BortoliMLP(mstd=self.marginal_prob_std_fn,encoder_layers=[16], pos_dim=16, decoder_layers=[128,128,128])
            mus=torch.as_tensor(np.array([0.,-5.,5.]))
            sigs=torch.as_tensor(np.array([1.,2.,3.]))
            weights=torch.ones((len(mus),),dtype=torch.float64)/len(mus)
            mu=np.dot(weights,mus)
            f=(mus**2+3*sigs**2)
            m4=torch.dot(weights,(f-2*mu*mus)**2+2*mu**2*f)-3*mu**4
            true_moments=np.array([mu,torch.dot(weights,mus**2+sigs**2)-mu**2,torch.dot(weights,(mus**2+3*sigs**2)*(mus-mu))+2*mu**3,m4])
        
            n=(batches*batch_size)//len(mus)
            x_train=torch.cat([mus[i]+sigs[i]*torch.randn(n,1) for i in range(len(mus))],0)
            datadist = lambda x: weights[None,:]@(norm.pdf(x,loc=mus[:,None],scale=sigs[:,None]))
            pdf = lambda vec,x,t: torch.sum(torch.as_tensor(norm.pdf(x,loc=mus*mean_map_fn(t),
                                         scale=torch.sqrt(marginal_prob_std_fn(t)+mean_map_fn(t)**2*sigs**2)))*vec,dim=-1,keepdim=True)
        
            def trueScore(x,t):
                return pdf(weights*(mus*mean_map_fn(t)-x)/(marginal_prob_std_fn(t)+mean_map_fn(t)**2*sigs**2),x,t)/pdf(weights,x,t)
            def trueProb(x,t):
                return pdf(weights,x,t)
        
        elif self.data=='Gaussian2d':
            mu0=torch.tensor([1.,1.])
            sig0=2*torch.tensor([1.,1.])
            batch_size = 1000;batches=50;
            
            mlp=BortoliMLP(mstd=self.marginal_prob_std_fn,encoder_layers=[16], pos_dim=16, decoder_layers=[128,128],x_dim=2)
            mus=mu0[None,...]
            sigs=sig0[None,...]
            n_epochs =  200
            x_train=mu0+sig0*torch.randn(batches*batch_size,2)
            true_moments=torch.cat([mus,sigs**2,torch.zeros_like(mus),3*sigs**4]).numpy()
        
            datadist = lambda x: torch.as_tensor(multivariate_normal.pdf(x,mean=mu0,cov=sig0**2))
            def trueScore(x,t):
                return -(x-mean_map_fn(t)*mu0)/(marginal_prob_std_fn(t)+mean_map_fn(t)**2*sig0**2)
            def trueProb(x,t):
                return torch.as_tensor(multivariate_normal.pdf(x,mean=mean_map_fn(t)*mu0,cov=marginal_prob_std_fn(t)+mean_map_fn(t)**2*sig0**2))
            
        elif self.data=='GMM2d':
            batch_size = 1000;batches=50;
            weights=torch.ones((5,))/5.
            mlp=BortoliMLP(mstd=self.marginal_prob_std_fn,x_dim=2)
            mus=torch.as_tensor([[0.,0.],[-5.,-5.],[5.,5.],[5.,-5.],[-5.,5.]])
            sigs=torch.vstack([1.*torch.ones(self.dimension),2.*torch.ones(self.dimension),3.*torch.ones(self.dimension),3.*torch.ones(self.dimension),2.*torch.ones(self.dimension)])
            n_epochs =  400
            n=(batches*batch_size)//5
            x_train=torch.cat([mus[i]+sigs[i]*torch.randn(n,self.dimension) for i in range(len(mus))],0)
            mu=weights[None,...]@mus
            f=(mus**2+3*sigs**2)
            m4=weights[None,...]@((f-2*mu*mus)**2+2*mu**2*f)-3*mu**4
            true_moments=my_round(torch.cat([mu,weights[None,...]@(mus**2+sigs**2)-mu**2,weights[None,...]@((mus**2+3*sigs**2)*(mus-mu))+2*mu**3,m4]).numpy(),figs=1)
        
            def datadist(x): 
                ddist=torch.zeros(x.shape[:-1])
                for m,s,w in zip(mus,sigs,weights):
                    ddist+=w*multivariate_normal.pdf(x,mean=m,cov=s**2)
                return ddist
            def pdf(vec,x,t):
                ddist=torch.zeros(x.shape[:-1])
                i=0
                for m,s in zip(mus,sigs):
                    c=(marginal_prob_std_fn(t)+mean_map_fn(t)**2*s**2)
                    ddist+=vec[i]*multivariate_normal.pdf(x,mean=m*mean_map_fn(t),cov=c)
                return ddist
            def GMMHelper(vec,x,t):
                grad=torch.zeros(x.shape)
                i=0
                for m,s in zip(mus,sigs):
                    c=(marginal_prob_std_fn(t)+mean_map_fn(t)**2*s**2)
                    grad+=vec[i]*multivariate_normal.pdf(x,mean=m*mean_map_fn(t),cov=c)[...,None]
                return grad
            def trueScore(x,t):
                return GMMHelper(weights[:,None,None,None]*((mus*mean_map_fn(t))[:,None,None,:]-x[None,...])/((marginal_prob_std_fn(t)+mean_map_fn(t)**2*sigs**2)[:,None,None,:]),x,t)/(pdf(weights,x,t)[...,None]+1e-12)
            def trueProb(x,t):
                return pdf(weights,x,t)
        
        else:# data=='Rosenbrock2d':
            batch_size = 1000;batches=50;
            mlp=BortoliMLP(mstd=self.marginal_prob_std_fn,x_dim=2)
            n_epochs =  400
            xs=np.sqrt(10.)*torch.randn(batches*batch_size,1)+1.
            ys=xs**2+torch.randn(batches*batch_size,1)/np.sqrt(10.)
            x_train=torch.cat([xs,ys],1)
            true_moments=np.array([[1.,11.],[10.,240.1],[0.,10400.],[300.,840000.]]) #844944.03 really
            mus=[None]
            sigs=[None]
            def datadist(x): 
                ddist=torch.exp(-(100.*(x[...,1]-x[...,0]**2)**2+(1.-x[...,0])**2)/20.)
                return .5*ddist/torch.pi
            
            def trueScore(x,t):
                return None
            def trueProb(x,t):
                return None
        
        self.datadist=datadist
        self.trueScore=trueScore
        self.trueProb=trueProb
        self.true_moments=true_moments
        self.norm_std,self.norm_mu=torch.std_mean(x_train, dim=0)
        if train:
            optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
            losses=[]
            if normalise:x_train=(x_train-self.norm_mu)/self.norm_std
            train_loader=torch.utils.data.DataLoader(dataset=x_train,batch_size=batch_size, shuffle=True)
            if __name__ == '__main__':
                tqdm_epoch = tqdm.trange(n_epochs)
                for epoch in tqdm_epoch:
                    avg_loss = 0.
                    num_items = 0
                    for x in train_loader:
                        loss = loss_function(mlp, x, self.marginal_prob_std_fn, self.mean_map_fn,self.diffcoeff)
                        optimizer.zero_grad()
                        loss.backward()    
                        optimizer.step()
                        avg_loss += loss.item() * x.shape[0]
                        num_items += x.shape[0]
                    # Print the averaged training loss so far.
                    l=avg_loss / num_items
                    losses+=[l]
                    tqdm_epoch.set_description('Average Loss: {:5f}'.format(l))
                    # Update the checkpoint after each epoch of training
                    torch.save(mlp.state_dict(), self.tag+'ckpt.pth')
        
            plt.plot(losses)
            plt.title('Loss Function')
        else:
            ckpt = torch.load(self.tag+'ckpt.pth')
            mlp.load_state_dict(ckpt)
        
        self.mlp=mlp
        self.mus=mus
        self.sigs=sigs
        return
#%%
for sde in ['VE']:
    # for data in ['Gaussian']:
    #     exp=Experiment(sde, data,dim=1,train=True,normalise=normalise)
    #     exp=Experiment(sde, data,dim=2,train=True,normalise=normalise)
    exp=Experiment(sde, 'Rosenbrock',dim=2,train=True)
#%%
def EM(y0,M,drift,diff,normalise=False,nstd=None,nmu=None):
    h=T/M
    N=len(y0)
    y=y0.detach().clone()
    with torch.no_grad():
        for m in range(M):
            t = torch.ones((N,1)) * (T-m*h)
            scale=diff(y,t)*np.sqrt(h)
            y+=drift(y, t)*h
            y+=torch.randn_like(y)*scale
    if normalise:
        y=nstd*y+nmu
    return y

#### Y ####
range_=torch.arange(0,M+1,1)/M
hrange=T*range_
times=np.array([.01,.2,.5,.8,1.])*T
def plotter(data,dim,sim,eqs=['Drift', 'VE', 'VP', 'subVP']):
    vmax=0
    if dim==2:
        for i,equation in enumerate(eqs):
            exp=Experiment(sde=equation, data=data,dim=dim,normalise=normalise)
            if i==0:
                nbins=100
                fig,ax=plt.subplots(1,5,figsize=(20,4),sharey=True)
                if exp.data=='Rosenbrock2d':
                    bins=[torch.linspace(-15, 15, nbins),torch.linspace(-20, 100, nbins)]
                elif exp.data=='Gaussian2d':
                    bins1 = torch.linspace(exp.mus[0,0]-3*exp.sigs[0,0], exp.mus[0,0]+3*exp.sigs[0,0], nbins).numpy()
                    bins2= torch.linspace(exp.mus[0,1]-3*exp.sigs[0,1], exp.mus[0,1]+3*exp.sigs[0,1], nbins).numpy()
                    bins=np.array([bins1,bins2])
                else:#exp.data=='GMM2d':
                     bins1 = torch.linspace(torch.min(exp.mus)-torch.max(exp.sigs), torch.max(exp.mus)+torch.max(exp.sigs), nbins).numpy()
                     bins=np.array([bins1,bins1])
                [xbins,ybins]=bins
                x=.5*(xbins[1:]+xbins[:-1])
                y=.5*(ybins[1:]+ybins[:-1])
                newx,newy=np.meshgrid(x,y)
                xmesh=torch.tensor(np.concatenate((newx[...,None],newy[...,None]),axis=-1))
                ddist=exp.datadist(xmesh)
                plt.figure(fig)
                problevels=np.linspace(0,torch.max(ddist),10)
                plot1=ax[4].contourf(newx,newy,ddist,levels=problevels)
                ax[4].set_title('True Dist.')
                plt.colorbar(plot1,ax=ax[4],label='Prob. Dens.')
                printer=[f'$M_{i+1}={np.array2string(exp.true_moments[i],separator=", ")}$\n' for i in range(4)]
                ax[4].annotate('\\underline{Moments}\n'+''.join(printer)[:-1],
                              xy=(.5,.8),fontsize=5,xycoords='axes fraction',bbox=dict(boxstyle="round,pad=0.3", fc='white',ec="k", lw=1))
                ax[4].set_xlabel('$x_0$')
            plt.figure(fig)
            ax[i].set_xlabel('$x_0$')
            tag=exp.tag
            driftcoeff,diffcoeff,marginal_prob_std=exp.driftcoeff,exp.diffcoeff,exp.marginal_prob_std_fn
            mlp = lambda x,t: exp.mlp(x,t)/marginal_prob_std(t)
            if sim:
                def MLPDrift(y,t):
                    return -driftcoeff(y,t)+mlp(y,t)*(diffcoeff(y, t)**2)
                yfin=EM(exp.y0,M,MLPDrift,diffcoeff,normalise,exp.norm_std,exp.norm_mu)
                if save:
                    with open(tag+".txt", "wb") as outf:
                        pickle.dump({'yfin':yfin}, outf) 
            else:
                with open(tag+".txt", "rb") as outf:
                    dictionary=pickle.load(outf)
                    yfin=dictionary['yfin']
            vals,[xbins,ybins]=np.histogramdd(np.array(yfin),bins=bins,density=True)
            vals=vals.T
            newx,newy=np.meshgrid(x,y)
            plot1=ax[i].contourf(newx,newy,vals,levels=problevels)
            # plt.colorbar(plot1,ax=ax[i])
            ax[i].set_title(equation+' MLP')
            m1,m2,m3,m4=summary_stats(yfin)
            ax[i].annotate('\\underline{Moments}\n'+f'$M_1={m1}$\n$M_2={m2}$\n$M_3={m3}$\n$M_4={m4}$', 
                         xy=(.5,.8),fontsize=5,xycoords='axes fraction',bbox=dict(boxstyle="round,pad=0.3", fc='white',ec="k", lw=1))
            ax[i].set_xlabel('$x_0$')
            
            if data!='Rosenbrock':
                #ScoreError
                fig2,ax2=plt.subplots(2,len(times),figsize=(4*len(times),6),sharey=True,sharex=True)
                plt.figure(fig2)
                ax2[0,0].set_ylabel('$x_1$');ax2[1,0].set_ylabel('$x_1$')
                for j,t in enumerate(times):
                    plottable1=exp.trueProb(xmesh,t)
                    plot1=ax2[0,j].contourf(newx,newy,plottable1,levels=10)
                    ax2[1,j].set_xlabel('$x_0$')
                    ax2[0,j].set_title(f'$t={t}$')
                    with torch.no_grad():
                        t_=torch.ones((xmesh.shape[0],xmesh.shape[1],1))*t
                        ts=exp.trueScore(xmesh,t) if normalise else exp.trueScore(xmesh,t)
                        plottable2=torch.linalg.vector_norm((mlp(xmesh,t_)-ts),dim=-1)/(torch.linalg.vector_norm(ts,dim=-1)+1e-10)
                    vmax=torch.sort(plottable2.flatten()).values[int(.95*len(plottable2.flatten()))]
                    plot2=ax2[1,j].contourf(newx,newy,torch.minimum(plottable2,vmax),levels=10)
                    if j==len(times)-1:
                        plt.colorbar(plot1,ax=ax2[0,j],label='True Prob. Dens.')
                        plt.colorbar(plot2,ax=ax2[1,j],label='Frac. SE: $\\vert\\vert \\frac{\\nabla\log P_t(x)-s_{\\theta}(x,t)\\vert\\vert _2}{\\vert\\vert\\nabla\log P_t(x)\\vert\\vert_2}$')
                    else:
                        plt.colorbar(plot1,ax=ax2[0,j])
                        plt.colorbar(plot2,ax=ax2[1,j])
                    error=torch.sum(plottable1*plottable2)/torch.sum(plottable1)
                    ax2[1,j].annotate('$\mathbb{E}[FSE]='+f'{round(error.item(),2)}$',xy=(.6,-.17),fontsize=8,xycoords='axes fraction',bbox=dict(boxstyle="round,pad=0.3", fc='white',ec="k", lw=1))
                plt.figure(fig2)
                fig2.suptitle(exp.data + ' : '+exp.title)
                plt.tight_layout(w_pad=.01,h_pad=.1)
                if save:
                    plt.savefig(exp.tag+'ScoreError.pdf', format='pdf', bbox_inches='tight')
        plt.figure(fig)
        ax[0].set_ylabel('$x_1$')
        fig.suptitle(exp.data)
        plt.tight_layout(w_pad=0.,h_pad=.01)
        if save:
            tag=exp.data+'Norm' if normalise else exp.data
            plt.savefig(tag+'Dist.pdf', format='pdf', bbox_inches='tight')
        # torch.sum(ddist[...,None]*(torch.tensor(np.concatenate((newx[...,None],newy[...,None]),axis=-1))),dim=(0,1))/torch.sum(ddist)
    else:
        for i,equation in enumerate(eqs):
            exp=Experiment(sde=equation, data=data,dim=dim,normalise=normalise)
            tag=exp.tag
            if i==0:
                fig=plt.figure()
                nbins=1000
                bins = torch.linspace(torch.min(exp.mus)-3*torch.max(exp.sigs), torch.max(exp.mus)+3*torch.max(exp.sigs), nbins)
                x=.5*(bins[1:]+bins[:-1])
                markers=['X','s','d','^']
                ddist=exp.datadist(x).flatten()
            driftcoeff,diffcoeff,marginal_prob_std=exp.driftcoeff,exp.diffcoeff,exp.marginal_prob_std_fn
            mlp=lambda x,t: exp.mlp(x,t)/marginal_prob_std(t)
            if sim:
                def MLPDrift(y,t):
                    return -driftcoeff(y,t)+mlp(y,t)*(diffcoeff(y, t)**2)
                yfin=EM(exp.y0,M,MLPDrift,diffcoeff,normalise,exp.norm_std,exp.norm_mu)
                vals,_=torch.histogram(yfin,bins=bins,density=True)
                if save:
                    with open(tag+".txt", "wb") as outf:
                        pickle.dump({'vals':vals,'x':x}, outf) 
            else:
                with open(tag+".txt", "rb") as outf:
                    dictionary=pickle.load(outf)
                    vals,x=dictionary['vals'],dictionary['x']
            plt.figure(fig)
            plt.plot(x,vals,'k-',label=equation+' MLP',alpha=.5,marker=markers[i],markevery=100)
            wdist=Wasserstein1(vals, ddist)
            plt.annotate(f'{equation} $W_1={round(wdist.item(),4)}$', xy=(.65,.7-.06*i),fontsize=8,xycoords='axes fraction',bbox=dict(boxstyle="round,pad=0.3", fc='white',ec="k", lw=1))
       
            
            xmesh = torch.linspace(-12, 12, 1000)
            t = torch.linspace(1e-5, T, 1000)
            xv, tv = torch.meshgrid(xmesh, t)
            
            plottable1=exp.trueProb(xv[...,None],tv[...,None])[...,0]
            fig2,ax=plt.subplots(1,2,figsize=(8,3),sharey=True)
            plt.figure(fig2)        
            plot1=ax[0].contourf(xv,tv,plottable1,levels=10)
            ax[0].set_ylabel('$t$')
            ax[0].set_xlabel('$x$')
            ax[1].set_xlabel('$x$')
            ax[0].set_title('$P_t(x)$, '+exp.data)
            ax[1].set_title('$\\vert \\nabla\log P_t(x)-s_{\\theta}(x,t)\\vert / \\vert \\nabla\log P_t(x)\\vert$')
            
            with torch.no_grad():
                ts=exp.trueScore(xv[...,None],tv[...,None])[...,0] if normalise else exp.trueScore(xv[...,None],tv[...,None])[...,0] 
                plottable2=torch.abs(mlp(xv[...,None],tv[...,None])[...,0]-ts)/(torch.abs(ts)+1e-10)
            vmax=torch.sort(plottable2.flatten()).values[int(.95*len(plottable2.flatten()))]
            plot2=ax[1].contourf(xv,tv,torch.minimum(plottable2,vmax),levels=10)
            plt.colorbar(plot2,ax=ax[1],label='$\\frac{\\vert \\nabla\log P_t(x)-s_{\\theta}(x,t)\\vert}{\\vert \\nabla\log P_t(x)\\vert}$')
            plt.colorbar(plot1,ax=ax[0],label='True Prob. Dens.')
            fig2.suptitle(exp.title.replace('\n',', '))
            plt.tight_layout(pad=.4)
            error=torch.sum(plottable1*plottable2)/torch.sum(plottable1)
            ax[1].annotate('$\mathbb{E}[FSE]='+f'{round(error.item(),2)}$',xy=(.6,-.17),fontsize=8,xycoords='axes fraction',bbox=dict(boxstyle="round,pad=0.3", fc='white',ec="k", lw=1))
            if save:
                plt.savefig(tag+'ScoreError.pdf', format='pdf', bbox_inches='tight')
                
        plt.figure(fig)        
        plt.plot(x, ddist, label='True Dist.',ls='--',color='k')
        plt.title('Distribution of Numerical $Y_h(T)$, '+exp.data)
        plt.legend()
        plt.ylabel('Prob. Density')
        plt.xlabel('$x$')
        if save:
            tag=exp.data+'Norm' if normalise else exp.data
            plt.savefig(data+'Dist.pdf', format='pdf', bbox_inches='tight')
#%%

for data in ['Gaussian','GMM']:
     for dim in [1,2]:
         plotter(data,dim,sim=True)
# plotter('Rosenbrock',2,sim=False)
#%%

###############################################################################
class GaussianFourierProjection(torch.nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = torch.nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
class SongMLP(torch.nn.Module):

    def __init__(self, decoder_layers=[256,128,128], x_dim=1, embed_dim=256):
        super().__init__()

        self.net = MLP(2 * embed_dim,
                       layer_widths=decoder_layers +[x_dim],
                       activate_final = False,
                       activation_fn=torch.nn.LeakyReLU())

        self.embed = torch.nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         torch.nn.Linear(embed_dim, embed_dim))

        self.x_encoder = MLP(x_dim,
                             layer_widths=[embed_dim//2,embed_dim],
                             activate_final = True,
                             activation_fn=torch.nn.LeakyReLU())
        
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        temb = self.act(self.embed(t))
        xemb = self.x_encoder(x)
        h = torch.cat([xemb ,temb], -1)
        out = self.net(h) 
        return out

class ScoreNet(torch.nn.Module):
    '''
    https://github.com/AntoineSalmona/Push-forward-Generative-Models/blob/main/dim1/models/networks.py
    '''
    def __init__(self,xdim=1):
        super().__init__()
        self.fc_1 = torch.nn.Linear(xdim,96)
        self.fc_2 = torch.nn.Linear(96+64,196)
        self.fc_3 = torch.nn.Linear(196+64,xdim)
        self.sigma_emb_1 = torch.nn.Linear(16,32)
        self.sigma_emb_2 = torch.nn.Linear(32,64)

    def forward(self,x,t):
        t = get_timestep_embedding(t,16)
        t = F.leaky_relu(self.sigma_emb_1(t),0.2)
        t =  F.leaky_relu(self.sigma_emb_2(t),0.2)
        x = F.leaky_relu(self.fc_1(x))
        x = torch.cat([x,t],-1)
        x = self.fc_2(x)
        x = torch.cat((x,t),-1)
        x = F.leaky_relu(x,0.2)
        return self.fc_3(x)

    
# =============================================================================
# if data=='Gaussian':
#     # mark=markers[i]
#     fig,ax=plt.subplots(nrows=2,ncols=1,sharex=True)
#     col='k'
#     ax[0].plot(hrange,means,col+'-',label=equation+' MLP')
#     # ax[0].plot([0],[meanx],col,marker='X',ms=ms)
#     ax[1].plot(hrange,stds,col+'-')
#     # ax[1].plot([0],[stdx],col,marker='X',ms=ms)
#     meany=meanY(hrange)
#     stdy=stdY(hrange)
#     ax[0].plot(hrange,meany,col,ls=':')
#     ax[1].plot(hrange,stdy,col,ls=':')
#     
#     
#     # ax[0].scatter([],[],color='k',marker='X',s=ms**2,label='$\mu_X(T)$')
#     # ax[1].scatter([],[],color='k',marker='X',s=ms**2,label='$\sigma_X(T)$')
#     ax[0].plot([],[],color='k',ls=':',label='$\mu_Y(t)$')
#     ax[1].plot([],[],color='k',ls=':',label='$\sigma_Y(t)$')
#     
#     # ax[0].set_ylim([0,2*(mu0+.1)]);ax[1].set_ylim([0,2*(sig0+.1)])
#     
#     ax[0].set_ylabel('$\mu$');ax[1].set_ylabel('$\sigma$')
#     ax[1].set_xlabel('$t$')
#     ax[0].axhline(y=mu0,label='$\mu_0$',ls='--',color='k')
#     ax[1].axhline(y=sig0,label='$\sigma_0$',ls='--',color='k')
#     ax[0].legend();ax[1].legend()
#     fig.suptitle(title+f', Cost = {M}')
#     
#     
#     ax[1].set_ylim([-.1,2*(sig0+.1)])
#     # plt.tight_layout()
#     
#     plt.savefig(equation+data+'MuSigma.pdf', format='pdf', bbox_inches='tight')
# 
# =============================================================================