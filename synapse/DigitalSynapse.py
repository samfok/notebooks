from pylab import *
from numpy import *

class NeuronSynapse:
    def __init__(self):
        self.u = 0           # u is the local synaptic state
        self.K = 5           # K is the spike multiplier (input)
        self.w = 2000        # w is the weight: scale factor
        self.syntau = 10000  # synapse tau, in microseconds
        
        self.dt = 100        # dt = integration time step, in microseconds
        self.updatetime = 0  # last update time
        
        self.syn = 0.0
        
        self.x = 0.0         # neuron state
        self.neutau = 5000   # time scale of neuron integration in microseconds
        self.threshold = 10  # spiking threshold for neuron state x
        
        self.digital = False
        
        # synapse digitization
        self.synibits = 4    # 4 bits for integer component
        self.synfbits = 12   # 12 bits for fractional component
        
        # neuron digitization
        self.neuibits = 4
        self.neufbits = 12
        
        self.stochastic = False
        
        # override all the soma computation, just use constant inputs
        self.override = False

        self.reset()
        
    # call to re-initialize
    def reset(self):
        self.u = 0
        self.updatetime = 0
        self.x = 0
        self.syn = 0
        self.pr = (self.dt+0.0)/(self.syntau)
        #self.pr = self.pr/(1+self.pr)
        
    def debug_syn(self,incur):
        self.override = True
        self.incur = incur
        
        
    def digital_mode(self,val):
        if val:
            self.digital = True
        else:
            self.digital = False
            
    def digitize (self,x,ibits,fbits):
        y = x*(1<<fbits)
        y = y % (1 << (fbits + ibits))
        return (int(y)+0.0)/(1<<fbits)
        
    def stochastic_mode(self,val):
        if val: 
            self.stochastic = True
        else: 
            self.stochastic = False
        
    def debug_stop(self):
        self.override = False
        
    # call if a spike arrives
    def inspike(self):
        if self.stochastic:
            self.u = self.u + self.K
        else:
            self.u = self.u + 1
    
    # run one time step.
    # return 1 if there is a spike, 0 if there isn't.
    def step(self,curtime):
        if self.updatetime > curtime: return 0

        self.updatetime = self.updatetime + self.dt
                
        # synaptic update
        if self.stochastic:
            for i in range(self.u):
                if random.uniform(0,1) < self.pr:
                    self.u = self.u-1
                
            incur = (self.u*self.w+0.0)/(0.0+self.K)*self.dt/(0.0+self.syntau)
            if self.digital:
                incur = self.digitize(incur, self.synibits, self.synfbits)
        else:
            self.syn = self.syn + \
                (self.u*self.w - self.syn)*(0.0+self.dt)/self.syntau
            self.u = 0
            incur = self.syn
            if self.digital:
                self.syn = self.digitize (self.syn,self.synibits,self.synfbits)
                #print 'old=', incur, '; new=', self.syn
                
        if self.override: incur = self.incur
            
        # neuron update
        self.x = self.x + \
            (0.0+self.dt)/self.neutau*(0.5*(self.x**2)-self.x+incur)
        
        if self.digital:
            self.x = self.digitize (self.x,self.neuibits,self.neufbits)
        
        if self.x > self.threshold: # spike detect
            self.x = 0.0
            return 1
        else:
            return 0

    def readtime(self):
        return self.updatetime - self.dt
    
    def readpostsyn(self):
        if self.stochastic:
            return (self.u*self.w)/(0.0+self.K)*self.dt/(0.0+self.syntau)
        else:
            return self.syn
    
    def readneuron(self):
        return self.x


class SpikeGen:
    def __init__(self):
        self.interval = 5000 # inter-spike interval (in microseconds)
        self.updatetime = 0  # next update time
        
    def reset(self):
        self.updatetime = 0
        
    def step(self,curtime):
        if curtime < self.updatetime: return 0
        self.updatetime = self.updatetime + self.interval

        return 1


t=0
r_total=range(int(1e5))
n_digital = NeuronSynapse()
n_analog = NeuronSynapse()
n_stochastic = NeuronSynapse()
g = SpikeGen()

g.interval = 2500

n_analog.dt = 4
n_digital.dt = 4
n_stochastic.dt = 4

n_analog.stochastic_mode(False)
n_analog.digital_mode(False)
n_analog.reset()

n_stochastic.K = 4
n_stochastic.stochastic_mode(True)
n_stochastic.digital_mode(True)
n_stochastic.reset()

n_digital.stochastic_mode(False)
n_digital.digital_mode(True)
n_digital.reset()

inspike = []

# analog 
outspike = []
incurrent = []
xvals = []

# digital
outspike2 = []
incurrent2 = []
xvals2 = []

# stochastic
outspike3 = []
incurrent3 = []
xvals3 = []

for i in r_total:
    r_total[i] = 1e-3*r_total[i]
    xvals.append(n_analog.readneuron())
    xvals2.append(n_digital.readneuron())
    xvals3.append(n_stochastic.readneuron())
    incurrent.append (n_analog.readpostsyn())
    incurrent2.append (n_digital.readpostsyn())
    incurrent3.append (n_stochastic.readpostsyn())
    if g.step(t) == 1: 
        inspike.append(t*1e-3)
        n_analog.inspike()
        n_digital.inspike()
        n_stochastic.inspike()
    if n_analog.step(t) == 1: outspike.append(t*1e-3)
    if n_digital.step(t) == 1: outspike2.append(t*1e-3)
    if n_stochastic.step(t) == 1: outspike3.append(t*1e-3)
    t = t + 1

subplots_adjust(hspace=0.8)
pl = subplot(4,1,1)
pl.set_color_cycle(['r','g','b'])
x0 = plot(r_total,xvals)
x1 = plot(r_total,xvals2)
x2 = plot(r_total,xvals3)
pl.legend((x0[0],x1[0],x2[0]), ('Analog', 'Digital', 'Stochastic'), loc=2)
title('Neuron x')
ax = subplot(4,1,2,sharex=pl)
ax.set_color_cycle(['r','g','b'])
plot(r_total,incurrent)
plot(r_total,incurrent2)
plot(r_total,incurrent3)
title('Synapse output')
px = subplot(4,1,3,sharex=pl)
px.set_color_cycle(['r','g','b'])
px.stem(inspike,[1]*len(inspike))
title ('Input spikes')
px = subplot(4,1,4, sharex=pl)
px.set_color_cycle(['r','g','b'])
if len(outspike) != 0: px.stem(outspike,[1]*len(outspike),linefmt='r',markerfmt='ro')
if len(outspike2) != 0: px.stem(outspike2,[0.75]*len(outspike2),linefmt='g',markerfmt='go')
if len(outspike3) != 0: px.stem(outspike3,[0.5]*len(outspike3),linefmt='b',markerfmt='bo')
title ('Output spikes')
show()


# do some fancy statistics

# def experiment(n,inspk,tm):
#     """ 
#     n :
#         neuron + synapse model, 
#     inspk :
#         spike generator, 
#     tm :
#         duration in usec
#     """
#     t=0
#     r_total=range(tm)
#     outcnt = 0
#     for i in r_total:
#         if inspk.step(tm) == 1:
#             n.inspike()
#         if n.step(tm) == 1: outcnt = outcnt + 1
#     return (0.0 + outcnt)/(0.0+tm)
# 
# 
# def many_exp(n,inspk,tm,count):
#     x = 0
#     x2 = 0
#     sys.stdout.write('Experiment ' + str(count) + ': ')
#     for i in range(count):
#         sys.stdout.write('[' + str(i) + ']')
#         sys.stdout.flush()
#         n.reset()
#         inspk.reset()
#         v = experiment(n,inspk,tm)
#         x = x + v
#         x2 = x2 + v*v
#     m = (x+0.0)/count
#     sys.stdout.write("\n")
#     return [m, math.sqrt(math.fabs((x2+0.0)/count-m**2))]

# n = NeuronSynapse()
# g = SpikeGen()

# orate = []
# ovar = []
# rate = np.linspace(0.5e3,0.7e3,20)
# #rate = [0.6e3]
# n.dt = 1
# n.K = 4
# n.stochastic_mode(False)
# n.digital_mode(False)
#  for i in rate:
#      print 'Processing ', i, '...'
#      n.reset()
#      g.reset()
#      g.interval = int(1e6/i)  # interval set by i, the input firing rate
#      l = many_exp(n,g,int(1e6),1)
#      orate.append(l[0]*1e6)
#      ovar.append(l[1]*1e6)

# stem(rate,orate)
# show()
