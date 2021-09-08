import numpy as np
import pandas as pd


def rate_nan_fixer(df, printer=False, NANGAPFILL=False):
    if NANGAPFILL:
        num_of_rows = len(df.index)
        list_of_rates = list(df["RATE"])
        list_of_times = list(df["START_TIME_DT"])
        column_index_of_rate = list(df.columns).index("RATE")
        for index in range(num_of_rows):
            rate = list_of_rates[index]
            if pd.isna(rate):
                curr_time_hour = list_of_times[index].hour
                prev_index = index - 1
                while list_of_times[prev_index].hour != curr_time_hour:
                    prev_index -= 1
                prev_rate = list_of_rates[prev_index]
                next_index = index + 1
                while list_of_times[next_index].hour != curr_time_hour:
                    next_index += 1
                next_rate = list_of_rates[next_index]
                if prev_rate == next_rate and not pd.isna(prev_rate):
                    if printer:
                        print(index)
                    df.iat[index, column_index_of_rate] = prev_rate
                else:
                    print("Don't know what to do for index {}".format(index))
                    break
    else:
        df=df.dropna()
    return df


def new_rate_indices_finder(df, return_rates=False):
    rate_list = list(df["RATE"])
    new_rate_indices = [0]
    new_rate = [rate_list[0]]
    for index in range(1, len(rate_list)):
        if rate_list[index] != rate_list[index - 1]:
            new_rate_indices.append(index)
            new_rate.append(rate_list[index])
    
    if return_rates:
        return new_rate_indices + [len(rate_list)], new_rate
    else:
        return new_rate_indices + [len(rate_list)]
    
    
def find_block_nums(df, street_name):
    block_nums = df.STREET_BLOCK.unique()
    return [int(name[len(street_name):]) for name in block_nums]


def dataset_creator_weekdays_window(street_name, timewin=(1400,1400), pre_loaded_dataset=None, daytype="weekday",NANGAPFILL=False):
    if pre_loaded_dataset is None:
        dataset = pd.read_csv('SFpark_ParkingSensorData_HourlyOccupancy_20112013.csv')
    else:
        dataset = pre_loaded_dataset
    
    assert street_name in dataset["STREET_BLOCK"].values, "Street name ({}) not found".format(street_name)
    dataset = dataset[dataset.STREET_BLOCK == street_name]
    dataset = dataset[dataset.DAY_TYPE == daytype]
    assert timewin[0] in dataset["TIME_OF_DAY"].values, "Time ({}) not found".format(street_name)
    dataset = dataset[(dataset.TIME_OF_DAY <= timewin[1]) & (dataset.TIME_OF_DAY>=timewin[0])]
    dataset["START_TIME_DT"] = pd.to_datetime(dataset["START_TIME_DT"])
    dataset = dataset.sort_values(by="START_TIME_DT")
    
    occupancy = dataset[["RATE", "START_TIME_DT", "TOTAL_TIME", "TOTAL_OCCUPIED_TIME"]]
    occupancy = occupancy.copy()
    occupancy["OCCUPANCY_FRAC"] = occupancy["TOTAL_OCCUPIED_TIME"] / occupancy["TOTAL_TIME"]
    occupancy = occupancy.set_index([pd.Index(list(range(len(occupancy.index))))])
    
    occupancy=rate_nan_fixer(occupancy, NANGAPFILL=NANGAPFILL)    # clean up the NaNs
    return occupancy

def dataset_creator_weekdays_window_multiblk(blks, timewin=(1400,1400), pre_loaded_dataset=None, NANGAPFILL=False):
    if pre_loaded_dataset is None:
        dataset_all = pd.read_csv('SFpark_ParkingSensorData_HourlyOccupancy_20112013.csv')
    else:
        dataset_all = pre_loaded_dataset
    blknames=["BEACH ST "+str(blk) for blk in blks]
    print(len(blknames))
    #assert street_name in dataset["STREET_BLOCK"].values, "Street name ({}) not found".format(street_name)
    dataset = dataset_all[dataset_all.STREET_BLOCK == blknames[0]]
    print(len(dataset))
    if len(blknames)>1:
        for blkname in blknames[1::]:
            print(blkname)
            dataset=pd.concat([dataset,q2[q2.STREET_BLOCK == blkname]])
            print(len(dataset))
        
    dataset = dataset[dataset.DAY_TYPE == "weekday"]
    #assert timewin[0] in dataset["TIME_OF_DAY"].values, "Time ({}) not found".format(street_name)
    dataset = dataset[(dataset.TIME_OF_DAY <= timewin[1]) & (dataset.TIME_OF_DAY>=timewin[0])]
    dataset["START_TIME_DT"] = pd.to_datetime(dataset["START_TIME_DT"])
    dataset = dataset.sort_values(by="START_TIME_DT")
    
    occupancy = dataset[["RATE", "START_TIME_DT", "TOTAL_TIME", "TOTAL_OCCUPIED_TIME"]]
    occupancy = occupancy.copy()
    occupancy["OCCUPANCY_FRAC"] = occupancy["TOTAL_OCCUPIED_TIME"] / occupancy["TOTAL_TIME"]
    occupancy = occupancy.set_index([pd.Index(list(range(len(occupancy.index))))])

    occupancy = rate_nan_fixer(occupancy, NANGAPFILL=NANGAPFILL)    # clean up the NaNs
    return occupancy


def average_occupancy_at_end(occ, num_of_days, num_of_hours_in_window, numdays=5):
    return np.mean(occ["OCCUPANCY_FRAC"].tail(num_of_days * num_of_hours_in_window))

def sf_park_price_trajectory(df, num_of_hours_in_window):
    rates = list(df["RATE"])[::num_of_hours_in_window * numdays]    # 5 weekdays
    price_trajectory = [(rates[0], 0)]
    for i in range(1, len(rates)-1):
        if rates[i] != price_trajectory[-1][0]:
            price_trajectory.append((rates[i], i))
    price_trajectory.append((rates[len(rates)-1], len(rates)-1))
    return price_trajectory


def grad_func_theta(m, gamma, occ, N=1e6):
    '''
    occ : occupancy data frame
    N   : number of samples (batch size)
    '''
    def grad_func(theta):
        samples = np.asarray(occ["OCCUPANCY_FRAC"].sample(int(n),replace=True))+m*theta
        samples = np.minimum(np.maximum(samples, np.zeros(np.size(samples))), np.ones(np.size(samples)))    # bound all entries between 0 and 1
        grad_log_p = (samples - 0.7) * m 
        return gamma * theta + 1/N * np.sum(grad_log_p)
    
    return grad_func

def gdupdate(theta, grad_func, eta=0.001, MAXITER=1000):
    thetas=[theta]
    eta = eta    # can change this initialization to something better
    for i in range(MAXITER): 
        thetas.append(min(max(-1, thetas[-1] - eta * grad_func(thetas[-1])), 1))

    return thetas

def grad_func_theta_q(m, gamma,d0,delta=0.8,n=int(1e4), N=int(1e6)):
    '''
       empirical gradient at current distribution d_t
    '''
    def grad_func_q(theta):
        samples = np.random.choice(d0,int(N), replace=True)+m*theta-delta**n*m*theta
        samples = np.minimum(np.maximum(samples, np.zeros(np.size(samples))), np.ones(np.size(samples)))    # bound all entries between 0 and 1
        grad_log_p = (samples - 0.7) * m *(1-delta**n)
        return gamma * theta + 1/N * np.sum(grad_log_p)
    
    return grad_func_q

def grad_func_theta_qt(m, gamma,dt, q_occ, delta=0.8,n=10000, N=int(1e6)):
    '''
       computes empirical gradient at current distribution d_t
       q_occ  : base distribution for D(\theta)
       dt     : current distribution d_t
    '''
    def grad_func_qt(theta):
        samples = (1-delta**n)*(np.random.choice(q_occ,int(N), replace=True)+m*theta)+delta**n*np.random.choice(dt,int(N), replace=True)
        samples = np.minimum(np.maximum(samples, np.zeros(np.size(samples))), np.ones(np.size(samples)))    # bound all entries between 0 and 1
        grad_ = (samples - 0.7) * m *(1-delta**n)
        return gamma * theta + 1/N * np.sum(grad_)
    
    return grad_func_qt


def estimate_occupancy(m, q, theta, N=1e4):
    d = np.asarray(pd.DataFrame.copy(q, deep=True)['OCCUPANCY_FRAC'])
    return np.mean(np.random.choice(d, int(N), replace=True)) + m*theta


def estimate_m_from_final_occupancy(q, theta, final_occ, N=1e4):
    d = np.asarray(pd.DataFrame.copy(q, deep=True)['OCCUPANCY_FRAC'])
    return (final_occ - np.mean(np.random.choice(d, int(N), replace=True))) / theta

def run_dynamics(d0,q, theta, n=int(1e4), delta=0.8 ):
    return delta**n*d0+(1-delta**n)*(q+m*theta)

class ddproblem:
    def __init__(self, delta, n, N, theta_init=0,Rmin=0,Rmax=8, gamma=1e-3, seeds=[1]):
        self.delta = delta
        self.n = n
        self.N = N
        self.gamma = gamma
        self.occ=None
        self.end_occ=None
        self.sf_park_prices=None
        self.sf_park_xcoords=None
        self.sf_park_ycoords=None
        self.m=-0.1
        self.M=0
        self.theta_init=theta_init
        self.Rmin=Rmin
        self.Rmax=Rmax
        self.seeds=seeds
        self.street_name=None
        self.blk=None
        self.q=None
        self.data_current_run_fo=None
        self.data_current_run_zo=None
        self.q_occ=None
        self.dim=1
        self.zo_avgs=None
        self.zo_stds=None
        self.zo_vars=None
        
    def run_dynamics(self, d0, theta, n):
        return self.delta**n*d0+(1-self.delta**n)*(self.q_occ+self.m*theta)
    
    def compute_occupancy(self,street_name, timewin=(1400,1400), pre_loaded_dataset=None, daytype="weekday",NANGAPFILL=False):
        if pre_loaded_dataset is None:
            dataset = pd.read_csv('SFpark_ParkingSensorData_HourlyOccupancy_20112013.csv')
        else:
            dataset = pre_loaded_dataset

        #assert street_name in dataset["STREET_BLOCK"].values, "Street name ({}) not found".format(street_name)
        dataset = dataset[dataset.STREET_BLOCK == street_name]
        dataset = dataset[dataset.DAY_TYPE == daytype]
        #assert timewin[0] in dataset["TIME_OF_DAY"].values, "Time ({}) not found".format(street_name)
        dataset = dataset[(dataset.TIME_OF_DAY <= timewin[1]) & (dataset.TIME_OF_DAY>=timewin[0])]
        dataset["START_TIME_DT"] = pd.to_datetime(dataset["START_TIME_DT"])
        dataset = dataset.sort_values(by="START_TIME_DT")

        occupancy = dataset[["RATE", "START_TIME_DT", "TOTAL_TIME", "TOTAL_OCCUPIED_TIME"]]
        occupancy = occupancy.copy()
        occupancy["OCCUPANCY_FRAC"] = occupancy["TOTAL_OCCUPIED_TIME"] / occupancy["TOTAL_TIME"]
        occupancy = occupancy.set_index([pd.Index(list(range(len(occupancy.index))))])

        occupancy=rate_nan_fixer(occupancy, NANGAPFILL=NANGAPFILL)    # clean up the NaNs
        return occupancy

    def setup(self, street_name, blk, timewin, pre_occ, starting_rate, num_days=5):
        '''
        return occupancy
        pre_occ: data frame with all parking data
        '''
        num_of_hours = (timewin[1] - timewin[0]) // 100 + 1
        self.occ = self.compute_occupancy(street_name + " " +str(blk), timewin=timewin, pre_loaded_dataset=pre_occ)
        self.end_occ = np.mean(self.occ["OCCUPANCY_FRAC"].tail(num_days * num_of_hours)) 
        self.blk=blk
        self.street_name=street_name
        rates = list(self.occ["RATE"])[::num_of_hours * num_days]    # 5 weekdays
        price_trajectory = [(rates[0], 0)]
        for i in range(1, len(rates)-1):
            if rates[i] != price_trajectory[-1][0]:
                price_trajectory.append((rates[i], i))
        price_trajectory.append((rates[len(rates)-1], len(rates)-1))
        self.sf_park_prices=price_trajectory
        
        self.sf_park_xcoords = [price[1] for price in self.sf_park_prices]
        self.sf_park_ycoords = [price[0] for price in self.sf_park_prices]
        
        self.q = self.occ[self.occ["RATE"]==starting_rate]
        self.q_occ=np.asarray(pd.DataFrame.copy(self.q, deep=True)['OCCUPANCY_FRAC'])
            
        self.m=((self.end_occ - np.mean(np.random.choice(np.asarray(self.q['OCCUPANCY_FRAC']), 
                    int(self.N), replace=True))) / (self.sf_park_ycoords[-1]-starting_rate))
        
        d0=np.asarray(pd.DataFrame.copy(self.q, deep=True)['OCCUPANCY_FRAC'])

        v=np.random.choice(d0,int(self.N), replace=True)
        self.theta_opt=-(np.mean(v)-0.7)*(self.m)/(self.gamma+(self.m)**2)

        self.M = self.gamma+(self.m*1)**2+np.mean(v)

        
    def _grad_func_theta_qt(self,dt,n):
        '''
           computes empirical gradient at current distribution d_t
           q_occ  : base distribution for D(\theta)
           dt     : current distribution d_t
        '''
        def grad_func_qt(theta):
            samples = ((1-self.delta**n)*(np.random.choice(self.q_occ,int(self.N), replace=True)
                                              +self.m*theta)+self.delta**n*np.random.choice(dt,int(self.N), replace=True))
            samples = np.minimum(np.maximum(samples, np.zeros(np.size(samples))), np.ones(np.size(samples)))    # bound all entries between 0 and 1
            grad_ = (samples - 0.7) * self.m *(1-self.delta**n)
            return self.gamma * theta + 1/self.N * np.sum(grad_)
    
        return grad_func_qt
    
    def runFOgrad(self,nT_pairs):
        alldata={}
        final_price_lst = []
        for seed in self.seeds:
            np.random.seed(seed)
            for p in nT_pairs:

                n=p[0]; T=p[1]

                thetas_alg=[self.theta_init]
                eta=1/(self.gamma+(self.m*1)**2)
                d0=np.asarray(pd.DataFrame.copy(self.q, deep=True)['OCCUPANCY_FRAC'])

                for t in range(T):
                    d0=self.run_dynamics(d0,thetas_alg[-1],n)
                    grad_func_qt_=self._grad_func_theta_qt(d0, n)
                    thetas_alg.append(min(max(self.Rmin, thetas_alg[-1] - eta * grad_func_qt_(thetas_alg[-1])), self.Rmax))

                alldata[p,seed]=thetas_alg
                
        self.data_current_run_fo=alldata
        
        
    def _loss_func_theta_qt(self,d0, n):
        def loss_func_qt(theta):
            samples = (1-self.delta**n)*(np.random.choice(self.q_occ,int(self.N), replace=True)+self.m*theta)+self.delta**n*np.random.choice(d0,int(self.N), replace=True)
            samples = np.minimum(np.maximum(samples, np.zeros(np.size(samples))), np.ones(np.size(samples)))    # bound all entries between 0 and 1
            loss_p = (samples - 0.7)**2
            return 0.5*(self.gamma * theta**2 + 1/self.N * np.sum(loss_p))

        return loss_func_qt


    

    def _sample_sphere(self,epsilon):
        """
        Returns a point on the sphere in R^d of radius epsilon
        """
        x = np.random.normal(size=self.dim)
        x /= np.linalg.norm(x)
        x *= epsilon
        return x

    def runZOgrad(self,nT_pairs, seeds=[1], VAREPSILON = 0.25, K=1):
        l_star=max([max([self.m*theta+qq for qq in self.q_occ]) for theta in np.linspace(self.Rmin,self.Rmax,1000)])
        alldata={}
        avgs={}
        stds={}
        vari={}
        for p in nT_pairs:
            temp=[]
            
            for seed in seeds:
                np.random.seed(seed)

                n=p[0]; T=p[1]

                thetas_alg=[self.theta_init]
                
                eta=1/(1*self.M*T**(1-3*VAREPSILON))
                mu=(self.dim**2*l_star**2 * eta / self.gamma) ** (1/4)
                
                d0=np.asarray(pd.DataFrame.copy(self.q, deep=True)['OCCUPANCY_FRAC'])
                for t in range(T):

                    v_t = self._sample_sphere(mu) #/(1*np.log(len(thetas_alg)+2)),1) #sample_spherical(1)
                    d0=self.run_dynamics(d0,thetas_alg[-1],n)
                    loss=self._loss_func_theta_qt(d0,n)
                    gradest_=loss(thetas_alg[-1]+v_t) * v_t
                    if K>1:
                        for k in range(K-1):
                            v_t = sample_sphere(mu) #sample_spherical(1)
                            gradest_+=loss(thetas_alg[-1]+v_t) * v_t
                    
                    gradest_=gradest_/K
                    
                    entry = min(max(self.Rmin, thetas_alg[-1] - eta * gradest_), self.Rmax)
                    if np.shape(entry) == (1,):
                        entry = entry[0]
                    thetas_alg.append(entry)    # unwrap the 1-D array to a scalar
                

                alldata[p,seed]=thetas_alg
                
            temp.append(np.asarray(alldata[p,seed]))
            
            temp = np.asarray(temp)
            #print(temp)
            #break
            #if p==(1,120):
            #    temp_=np.copy(temp)
            avgs[p]=np.mean(temp,axis=0)
            stds[p]=np.std(temp,axis=1)
            vari[p]=np.var(temp,axis=1)
            #return temp_
        self.data_current_run_zo=alldata
        self.zo_avgs=avgs
        self.zo_vars=vari
        self.zo_stds=stds
        
