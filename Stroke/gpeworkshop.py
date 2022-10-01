import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import GPy
from sklearn.model_selection import train_test_split
from SALib.sample import saltelli
from SALib.analyze import sobol
import seaborn as sns
import shutil
cmap=sns.cubehelix_palette(8,start=.5,rot=.75,hue=0,as_cmap=1)
plt.close("all")
outnames=['minu','meanu','maxu','minuaca','meanuaca','maxuaca','minp','meanp','maxp','minpaca','meanpaca','maxpaca','PI','PIaca','PPI','PPIaca']
init = os.getcwd()


def gpe(network,activity):
	print('### Import files ###')                        # Scaling factor for cardiac output
	netact = network+'_'+activity
	os.chdir('training')
	INP = np.loadtxt(network+'.INP')
	OUT = np.loadtxt(network+'.OUT')
	nnames = open('names.txt').read().splitlines()
	print(network)
		
	# split training-validation
	train_size = int(INP.shape[0]*0.8)
	val_size = INP.shape[0]-train_size
	
	# normalisation is not needed
	INPtrain, INPval, OUTtrain, OUTval = train_test_split(INP,OUT,test_size=0.2)
	
	### Emulator training here
	nvar=INPtrain.shape[1]
	ker=GPy.kern.RBF(nvar)#+GPy.kern.Matern52(nvar,ARD=True)
	print('emulator training')
	m1=GPy.models.GPRegression(INPtrain,OUTtrain,ker)
	m1.optimize()
	m1.optimize_restarts(num_restarts=3)
	yy,vv=m1.predict(INPval)
	plt.figure()
	plt.plot(OUTval,OUTval,'r',alpha=0.2)
	plt.plot(OUTval,yy,'k.',alpha=0.2)
	plt.title(network)
	plt.xlabel('Simulated output')
	plt.ylabel('Emulated output')
	return m1,nnames

M2,nnames=gpe('M2stroke','A')
os.chdir(init)
M3,nnames=gpe('M3stroke','A')
os.chdir(init)

### Local sensitivity: question 1
# set all parameters to 1
# sweep r3
xr3=np.ones([40,len(nnames)])
xr3[:,nnames.index('r3')]=np.linspace(0.6,1.4,40)
y2r3,v2r3=M2.predict(xr3)
y3r3,v3r3=M3.predict(xr3)

# set all parameters to 1
# sweep r5
xr5=np.ones([40,len(nnames)])
xr5[:,nnames.index('r5')]=np.linspace(0.6,1.4,40)
y2r5,v2r5=M2.predict(xr5)
y3r5,v3r5=M3.predict(xr5)


plt.figure()
plt.plot(xr3[:,nnames.index('r3')],y2r3[:,outnames.index('meanp')],'b',label='Stroke M2, variable = r3')
plt.plot(xr5[:,nnames.index('r5')],y2r5[:,outnames.index('meanp')],'b.-',label='Stroke M2, variable = r5')
plt.plot(xr3[:,nnames.index('r3')],y3r3[:,outnames.index('meanp')],'r',label='Stroke M3, variable = r3')
plt.plot(xr5[:,nnames.index('r5')],y3r5[:,outnames.index('meanp')],'r.-',label='Stroke M3, variable = r5')

plt.ylabel('Normalised meanp')
plt.xlabel('radius')
plt.title('Effect of varying r3 and r5 separately')
plt.legend()

### Local sensitivity: question 2

# set all parameters to 1.3
# sweep r5
xxr5=np.ones([40,len(nnames)])*1.3
xxr5[:,nnames.index('r5')]=np.linspace(0.6,1.4,40)
yy2r5,vv2r5=M2.predict(xxr5)
yy3r5,vv3r5=M3.predict(xxr5)

# set all parameters to 1.3
# set mysterious parameters to 1
# sweep r5
xm=np.ones([40,len(nnames)])*1.3
xm[:,nnames.index('Rt1')]=1
xm[:,nnames.index('Rt2')]=1
xm[:,nnames.index('r5')]=np.linspace(0.6,1.4,40)
ym,vm=M2.predict(xm)

plt.figure()
plt.plot(xr5[:,nnames.index('r5')],y2r5[:,outnames.index('meanp')],'b.-',label='Stroke M2, all parameters to 1 except r5')
plt.plot(xr5[:,nnames.index('r5')],yy2r5[:,outnames.index('meanp')],'bo-',label='Stroke M2, all parameters to 1.3 except r5')
plt.plot(xr5[:,nnames.index('r5')],ym[:,outnames.index('meanp')],'g',label='Stroke M2, mysterious parameters to 1')
plt.plot(xr5[:,nnames.index('r5')],y3r5[:,outnames.index('meanp')],'r.-',label='Stroke M3, all parameters to 1 except r5')
plt.plot(xr5[:,nnames.index('r5')],yy3r5[:,outnames.index('meanp')],'ro-',label='Stroke M3, all parameters to 1.3 except r5')
plt.ylabel('Normalised meanp')
plt.xlabel('radius')
plt.title('Effect of varying r5')
plt.legend()




### GLOBAL SENSITIVITY ANALYSIS

uncertainty=0.4
bounds = np.column_stack([np.ones(len(nnames))*(1-uncertainty),np.ones(len(nnames))*(1+uncertainty)]) # vary within +/- uncertainty
bounds[0,:]=[0.5,2.]    # force length uncertaintys

problem={'num_vars':len(nnames),'names':nnames,'bounds':bounds}
param_values = saltelli.sample(problem,1024,calc_second_order=False)

# Generate points for sensitivity analysis
yM2,vM2=M2.predict(param_values)
yM3,vM3=M3.predict(param_values)
si2=[]
si3=[]
for i in range(yM2.shape[1]):
	Si2=sobol.analyze(problem,yM2[:,i],calc_second_order=False)
	Si3=sobol.analyze(problem,yM3[:,i],calc_second_order=False)
	si2.append(Si2['S1'])
	si3.append(Si3['S1'])
	print(str(i+1))
si2=np.array(si2)
si3=np.array(si3)


plt.figure()
sns.heatmap(si2,cmap=cmap,xticklabels=nnames,yticklabels=outnames)
plt.title('M2stroke')

plt.figure()
sns.heatmap(si3,cmap=cmap,xticklabels=nnames,yticklabels=outnames)
plt.title('M3stroke')




### REDUCTION OF UNCERTAINTY
# Reduce uncertainty on r5 and run global sensitivity again

uncertainty=0.4
boundsred = np.column_stack([np.ones(len(nnames))*(1-uncertainty),np.ones(len(nnames))*(1+uncertainty)]) # vary within +/- uncertainty
boundsred[0,:]=[0.5,2.]    # force length uncertaintys
# reduce uncertainty on r5 to 0.05
reduction = 0.1
boundsred[nnames.index('r5'),:] = np.array([1.2-reduction,1.2+reduction])

problemred={'num_vars':len(nnames),'names':nnames,'bounds':boundsred}
param_valuesred = saltelli.sample(problemred,1024,calc_second_order=False)

# Generate points for sensitivity analysis
yM2red,vM2red=M2.predict(param_valuesred)
yM3red,vM3red=M3.predict(param_valuesred)
si2red=[]
si3red=[]
for i in range(yM2red.shape[1]):
	Si2=sobol.analyze(problem,yM2red[:,i],calc_second_order=False)
	Si3=sobol.analyze(problem,yM3red[:,i],calc_second_order=False)
	si2red.append(Si2['S1'])
	si3red.append(Si3['S1'])
	print(str(i+1))
si2red=np.array(si2red)
si3red=np.array(si3red)


plt.figure()
sns.heatmap(si2red,cmap=cmap,xticklabels=nnames,yticklabels=outnames)
plt.title('M2stroke, reduced uncertainty on r5')

plt.figure()
sns.heatmap(si3red,cmap=cmap,xticklabels=nnames,yticklabels=outnames)
plt.title('M3stroke, reduced uncertainty on r5')




# How does this affect the local sensitivity on M3 (question 2)?


plt.show()
print(ccc)

