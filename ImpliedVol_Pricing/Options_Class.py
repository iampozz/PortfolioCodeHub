import numpy as np
import scipy.stats as sp

class EU_Options_binomial:
    def __init__(self, S0, K, sigma, T, r, nodes, Call, european):
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.T = T
        self.r = r
        self.Call = True
        self.european = True
        self.nodes = nodes
        
    def EU_bin_pricing(self):
         
        delta_t = self.T / self.nodes
         
        u = np.exp(self.sigma * np.sqrt(delta_t))
        d = np.exp(-self.sigma * np.sqrt(delta_t))
         
        p = (np.exp(self.r * delta_t) - d) / (u - d)
         
        s_tree = np.zeros((self.nodes + 1, self.nodes + 1))
        f_tree = np.zeros((self.nodes + 1, self.nodes + 1))
        if self.Call == True and self.european == True:
             
            for x in range(0, self.nodes + 1):
                for j in range(0, x + 1):
                    s_tree[x,j] = self.S0 * u ** j * d ** (x - j)

             
            for j in range(0, self.nodes + 1):
                f_tree[self.nodes, j] = max(s_tree[self.nodes, j] - self.K,0)
                 
                     
            for x in range(self.nodes - 1, -1, -1):
                for j in range(0, x + 1):
                    f_tree[x,j] = np.exp(-self.r * delta_t) * (p * f_tree[x + 1, j + 1] + (1 - p) * f_tree[x + 1, j])
        
        if self.Call == False and self.european == True:
            
            for x in range(0, self.nodes + 1):
                for j in range(0, x + 1):
                    s_tree[x,j] = self.S0 * u ** j * d ** (x - j)

            
            for j in range(0, self.nodes + 1):
                f_tree[self.nodes, j] = max(self.K - s_tree[self.nodes, j],0)
                
                    
            for x in range(self.nodes - 1, -1, -1):
                for j in range(0, x + 1):
                    f_tree[x,j] = np.exp(-self.r * delta_t) * (p * f_tree[x + 1, j + 1] + (1 - p) * f_tree[x + 1, j])
        
        return s_tree, f_tree

class US_Options_binomial:
    def __init__(self, S0, K, sigma, T, r, nodes, Call, european):
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.T = T
        self.r = r
        self.Call = True
        self.european = False
        self.nodes = nodes    
    
    def US_bin_pricing(self):
        
        delta_t = self.T / self.nodes
        
        u = np.exp(self.sigma * np.sqrt(delta_t))
        d = np.exp(-self.sigma * np.sqrt(delta_t))
        
        p = (np.exp(self.r * delta_t) - d) / (u - d)

        s_tree = np.zeros((self.nodes + 1, self.nodes + 1))
        f_tree = np.zeros((self.nodes + 1, self.nodes + 1))
        
        if self.Call== True and self.european == False:
            for x in range(0, self.nodes + 1):
                for j in range(0, x + 1):
                    s_tree[x,j] = self.S0 * u ** j * d ** (x - j)

            
            for j in range(0, self.nodes + 1):
                f_tree[self.nodes, j] = max(s_tree[self.nodes, j] - self.K,0)
                
                    
            for x in range(self.nodes - 1, -1, -1):
                for j in range(0, x + 1):
                    f_tree[x,j] = max(np.exp(-self.r * delta_t) * (p * f_tree[x + 1, j + 1] + (1 - p) * f_tree[x + 1, j]),max(s_tree[x, j] - self.K,0) )
        
        if self.Call== False and self.european == False:
            for x in range(0, self.nodes + 1):
                for j in range(0, x + 1):
                    s_tree[x,j] = self.S0 * u ** j * d ** (x - j)

            
            for j in range(0, self.nodes + 1):
                f_tree[self.nodes, j] = max(self.K - s_tree[self.nodes, j],0)
                
                    
            for x in range(self.nodes - 1, -1, -1):
                for j in range(0, x + 1):
                    f_tree[x,j] = max(np.exp(-self.r * delta_t) * (p * f_tree[x + 1, j + 1] + (1 - p) * f_tree[x + 1, j]),max(self.K - s_tree[x, j],0) )
                    
        return s_tree, f_tree
    
class EU_BSM:
    def __init__(self, S0, K, sigma, T, r, Call, european):
         self.S0 = S0
         self.K = K
         self.sigma = sigma
         self.T = T
         self.r = r
         self.Call = True
         self.european = True
    
    def BSM(self):
        
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * self.T ) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        if self.Call == True:
            c = self.S0 * sp.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * sp.norm.cdf(d2)
            
            return c
        if self.Call == False:
            p = self.K * np.exp(-self.r * self.T) * sp.norm.cdf(-d2) - self.S0 * sp.norm.cdf(-d1)
            
            return p
            
EU_opt = EU_Options_binomial(100, 105, 0.3, 1, 0.07, 252, Call = False, european = True)
US_opt = US_Options_binomial(100, 105, 0.3, 1, 0.07, 252, Call = False, european = False)

EU_price = EU_opt.EU_bin_pricing()
US_price = US_opt.US_bin_pricing()

BSM_opt_c = EU_BSM(100, 105, 0.3, 1, 0.07, True, True)
BSM_opt_p = EU_BSM(100, 105, 0.3, 1, 0.07, False, True)

BSM_c_price = BSM_opt_c.BSM()
BSM_p_price = BSM_opt_p.BSM()