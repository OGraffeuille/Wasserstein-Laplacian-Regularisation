# Imports
import os
import time
import math
import random
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt  

# Sklearn
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim

# Our scripts
from src.utils import *

# Set up torch for cuda
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MDN network
class MDN_nn(nn.Module):

    def __init__(self, input_dims, num_hidden, num_gaussians):
        super(MDN_nn, self).__init__()
        
        self.input_dims = input_dims
        self.num_gaussians = num_gaussians
        self.num_hidden = num_hidden
        
        linear_layers = [nn.Linear(input_dims, num_hidden[0])]
        if len(num_hidden) > 1:
            for i in range(1, len(num_hidden)):
                linear_layers.append(nn.Linear(num_hidden[i-1], num_hidden[i]))
                
        self.nn_h = nn.ModuleList(linear_layers)
        
        self.nn_pi = nn.Linear(num_hidden[-1], num_gaussians)
        self.nn_sigma = nn.Linear(num_hidden[-1], num_gaussians)
        self.nn_mu = nn.Linear(num_hidden[-1], num_gaussians)

    def forward(self, x):
        
        for nn_h in self.nn_h:
            x = F.relu(nn_h(x))
            
        sigma = torch.exp(self.nn_sigma(x))
        mu = self.nn_mu(x)
        pi = F.softmax(self.nn_pi(x), dim=1)
        
        return mu, sigma, pi
    
    
    
class WMDN(object):
    
    # Initialise the MDN architecture
    #                    X, y - (nxm, 1xm) np matrices containing data and labels
    #              l, u, v, t - number of labelled, unlabelled, validation, test data points
    #              input_dims - number of features (number of inputs to MDN architecture)
    #              num_hidden - array of integers (e.g. [32, 32, 32]), number of neurons on each hidden layer
    #           num_gaussians - number of gaussians to parameterise with MDN (c)
    #                    seed - seed used to sample labelled/unlabelled datasets, initialise MDN weights, etc
    #                      lr - model learning rate
    #     batch_per_epoch_NLL - number of batches per epoch for NLL loss
    #     batch_per_epoch_WLR - number of batches per epoch for WLR loss (generally 1)
    #                patience - number of epochs with increasing validation loss required to stop training
    #              epochs_max - maximum number of epochs
    #       validation_epochs - how frequently to check validation loss
    #             time_epochs - whether to time and print duration of each epoch
    #              ssl_weight - gamma_u, weighting parameter to WLR loss
    #                       b - number of bins used in WLR computation
    #              sigma_coef - used to multiple default sigma value
    #                       d - RBF kernel power
    #                       q - WLR power coefficient, analogous to L_p loss
    #                       k - kNN parameter for WLR kernels  
    def fit(self, X, y, l, u, v, t, \
                  input_dims, num_hidden, num_gaussians=5, \
                  seed=0, lr=1e-3, batch_per_epoch_NLL=10, batch_per_epoch_WLR=1, patience=20, \
                  epochs_max=5000, validation_epochs=10, time_epochs=False, \
                  ssl_weight=1, b=20, sigma_coef=1, d=1, q=2, k=5):
        
        # Architecture parameters
        self.input_dims = input_dims
        self.num_hidden = num_hidden
        self.num_gaussians = num_gaussians
        
        # Training parameters
        self.seed = seed
        self.lr = lr
        self.batch_per_epoch_NLL = batch_per_epoch_NLL
        self.batch_per_epoch_WLR = batch_per_epoch_WLR
        self.patience = patience
        self.epochs_max = epochs_max
        self.validation_epochs = validation_epochs
        self.time_epochs = time_epochs
        
        # SSL training parameters
        self.ssl_weight = ssl_weight
        self.b = b
        self.d = d
        self.k = k
        self.q = q
        self.sigma_coef = sigma_coef
            
        # Determine whether to run algorithm as semi-supervised or supervised
        self.is_ssl = (self.ssl_weight > 0) and (self.k > 0)
        
        # Set seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Normalise and sample data
        self.X, self.y, self.X_mean, self.y_mean, self.X_std, self.y_std = normalise_data(X, y)
        self.X_l, self.y_l, self.X_u, self.y_u, self.X_v, self.y_v, self.X_t, self.y_t = \
            split_data(self.X, self.y, l, u, v, t, test_state=0, label_state=self.seed)        
        
        # Check range of data for Wasserstein calculation
        self.y_min = np.min(self.y_l)
        self.y_max = np.max(self.y_l)
                 
        # Create tensor objects for each dataset
        self.X_l_ten = torch.from_numpy(self.X_l).to(device).float()
        self.y_l_ten = torch.from_numpy(self.y_l).to(device).float()
        self.X_u_ten = torch.from_numpy(self.X_u).to(device).float()
        self.y_u_ten = torch.from_numpy(self.y_u).to(device).float()
        self.X_v_ten = torch.from_numpy(self.X_v).to(device).float()
        self.y_v_ten = torch.from_numpy(self.y_v).to(device).float()
        self.X_t_ten = torch.from_numpy(self.X_t).to(device).float()
        self.y_t_ten = torch.from_numpy(self.y_t).to(device).float()
        
        # Create joint training set objects objects
        self.X_lu = np.concatenate([self.X_l, self.X_u])
        self.X_lu_ten = torch.from_numpy(self.X_lu).to(device).float()
        self.X_lu_inds_ten = torch.arange(l+u, dtype=torch.int64).view(-1,1)
        
        # Define dataset objects
        self.dataset_train_sup = torch.utils.data.TensorDataset(self.X_l_ten, self.y_l_ten)
        self.dataset_train_ssl = torch.utils.data.TensorDataset(self.X_lu_ten, self.X_lu_inds_ten)
        
        # Adjust parameters
        self.batch_size_sup = math.ceil(self.X_l_ten.shape[0] / self.batch_per_epoch_NLL)
        self.batch_size_ssl = math.ceil(self.X_lu_ten.shape[0] / self.batch_per_epoch_WLR)
        self.single_ssl_batch = self.batch_per_epoch_WLR <= 1
        self.kernel_sigma = self.get_sigma()
        self.patience = math.ceil(self.patience / self.validation_epochs)
        
        # Warnings
        if self.X_l_ten.shape[0] % self.batch_size_sup != 0:
            print("Warning: there will be a smaller supervised batch at the end of each Epoch.")
        if self.X_lu_ten.shape[0] % self.batch_size_ssl != 0:
            print("Warning: there will be a smaller semisupervised batch at the end of each Epoch.")
        if self.is_ssl and l > u:
            print("Warning: there are more labelled than unlabelled data.")
                  
        # Define data loader objects
        self.loader_train_sup = torch.utils.data.DataLoader(self.dataset_train_sup, batch_size=self.batch_size_sup, shuffle=True)
        self.loader_train_ssl = torch.utils.data.DataLoader(self.dataset_train_ssl, batch_size=self.batch_size_ssl, shuffle=not self.single_ssl_batch)
        
        # Create model and optimiser
        self.model = MDN_nn(self.input_dims, self.num_hidden, self.num_gaussians)
        optimiser = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Variable used to time execution
        epoch_start_time = time.time() 
        
        # Create lists to record losses
        self.train_nll = []
        self.train_wlr = []
        self.train_sum = []
        self.v_nlls = []
        self.v_r2s = []
        self.v_mses = []
        for self.epoch in range(self.epochs_max):
            
            epoch_nll_loss = 0
            epoch_wlr_loss = 0
            
            for X_batch, y_batch in self.loader_train_sup:
                
                # Make prediction
                mu, sigma, pi = self.model(X_batch)
                
                # Compute log loss
                nll_loss_new = self.gaussian_mixture_NLL(y_batch, mu, sigma, pi)
                
                # Normalise log loss by fraction of total log loss for that epoch
                len_batch = X_batch.shape[0]
                nll_loss = nll_loss_new * len_batch / l
                
                # Back propogation
                optimiser.zero_grad()
                nll_loss.backward()
                optimiser.step()
                
                epoch_nll_loss += nll_loss.item()
                
            
            if not self.is_ssl:
                epoch_wlr_loss.append(0)
            else:
                for X_batch, inds_batch in self.loader_train_ssl:
                    
                    # Make prediction
                    mu, sigma, pi = self.model(X_batch)
                    
                    # Compute SSL loss
                    wlr_loss = self.WLR_loss(mu, sigma, pi, inds_ten=inds_batch)
                    
                    # Normalise log loss by fraction of total log loss for that epoch
                    len_batch = X_batch.shape[0]
                    wlr_loss = wlr_loss * len_batch / (l + u)
                    
                    # Back propogation
                    optimiser.zero_grad()
                    wlr_loss.backward()
                    optimiser.step()      
                    
                    epoch_wlr_loss += wlr_loss.item()
                 
               
            # Check on validation every n epochs
            if self.epoch % self.validation_epochs == 0:
                
                # Record training metrics - assumes all batches are equal sizes
                self.train_nll.append(epoch_nll_loss)
                self.train_wlr.append(epoch_wlr_loss)
                self.train_sum.append(epoch_nll_loss + epoch_wlr_loss)
                
                # Record validation metrics
                v_metrics = self.get_metrics(self.X_v_ten, self.y_v_ten, verbose=False)
                self.v_nlls.append(v_metrics["nll"])
                self.v_r2s.append(v_metrics["r2_mode"])
                self.v_mses.append(v_metrics["mse_mode"])
                
                print("Epoch {}: NLL_loss {:5f}, WLR_loss {:5f}, v_NLL_loss {:5f}, v_MSE {:5f}, v_R2 {:5f}".format( \
                    self.epoch, self.train_nll[-1], self.train_wlr[-1], v_metrics["nll"], v_metrics["mse_mode"], v_metrics["r2_mode"]))
                
                # Validation stopping check
                if len(self.v_nlls) > self.patience and np.mean(np.array(self.v_nlls)[-self.patience-1:-1]) < self.v_nlls[-1]:
                    print("Validation threshold reached - stopping")
                    break
                        
            if self.time_epochs:
                print("Epoch took: {:1f}s".format(time.time() - epoch_start_time))
                epoch_start_time = time.time()
                        
        
    def plot_results(self, X_ten=None, y_ten=None, dim=0):
        
        if X_ten == None or y_ten == None:
            X_ten = self.X_t_ten
            y_ten = self.y_t_ten
        
        # Plot resulting GMM
        with torch.no_grad():
            
            X = X_ten.detach().cpu().numpy()
            y = y_ten.detach().cpu().numpy()
            
            mu, sigma, pi = self.model(X_ten)
            pred = self.predict(X_ten).reshape(-1, 1)
            y = y.reshape(-1, 1)
            
        # Determine X dimension to plot
        X_axis = X[:,dim]

        # Scatter plot transparency
        a = np.minimum(1, 50/len(y))

        # Plotting variables
        w = pi.cpu().numpy()
        means = mu.cpu().numpy()
        sigma = sigma.cpu().numpy()
        upper = means + sigma
        lower = means - sigma

        # Normalise variables
        means = means * self.y_std + self.y_mean
        sigma = sigma * self.y_std + self.y_mean
        upper = upper * self.y_std + self.y_mean
        lower = lower * self.y_std + self.y_mean
        pred = pred * self.y_std + self.y_mean
        y = y * self.y_std + self.y_mean
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot learning curve
        axs.flat[0].set_title("Loss curve")
        epochs = np.arange(len(self.train_nll))*self.validation_epochs
        axs.flat[0].plot(epochs, self.train_nll, label="NLL loss")
        axs.flat[0].plot(epochs, self.train_wlr, label="WLR loss")
        axs.flat[0].plot(epochs, self.train_sum, label="total loss")
        axs.flat[0].plot(epochs, self.v_nlls, label="validation loss")
        axs.flat[0].legend()
        
        # Plot prediction vs. real
        axs.flat[1].set_title("Predictions vs Real")
        axs.flat[1].set_xlabel("Real")
        axs.flat[1].set_ylabel("Predictions")
        axs.flat[1].scatter(y, pred, alpha=a, label="pred")
        min_val = min(y.min(), pred.min())
        max_val = max(y.max(), pred.max())
        axs.flat[1].plot([min_val, max_val], [min_val, max_val], ":")
        axs.flat[1].legend()
        
        # Gaussians learnt
        axs.flat[2].set_title("Gaussians")
        axs.flat[2].scatter(X_axis, y, alpha=a)
        for i in range(self.num_gaussians):
            axs.flat[2].scatter(X_axis, means[:,i].flatten(), s=w[:,i]*50, label="means"+str(i))
            axs.flat[2].scatter(X_axis, upper[:,i].flatten(), s=5, c="k", alpha=a)
            axs.flat[2].scatter(X_axis, lower[:,i].flatten(), s=5, c="k", alpha=a)
        axs.flat[2].legend()
            
        plt.show()
        
        
    def predict(self, X_ten, method="mean"):
        
        with torch.no_grad():
            mu, sigma, pi = self.model(X_ten)
            
        preds_means, preds_MLC, preds_mode = self.find_dist_estimates(mu, sigma, pi)

        if method == "mean":
            return preds_mode
        elif method == "MLC":
            return preds_MLC
        elif method == "mode":
            return preds_mode
        
        
    def find_dist_estimates(self, mu, sigma, pi, n_bins=1001):
        
        # Return mean of gaussian if mixture has just one component
        if mu.shape[1] == 1:
            y_opt = mu[:, 0]
        else:
            
            # Returns mean of mixture distribution
            weighted_means = pi * mu
            means = torch.sum(weighted_means, axis=1).detach().cpu().numpy()
            
            # Returns mean of most likely component of distribution (mu of distribution with greatest pi)
            inds = torch.argmax(pi, axis=1).view(-1,1)
            MLCs = torch.gather(mu, 1, inds) .detach().cpu().numpy()

            # Returns mode of mixture distribution, estimated with bins
            gmm_bins = self.get_bins(n_bins)
            gmm_pdf = self.gaussian_mixture_pdf(mu, sigma, pi, gmm_bins)
            
            ind_modes = torch.argmax(gmm_pdf, axis=1)
            modes = gmm_bins[ind_modes].detach().cpu().numpy()
            
        # Convert before returning estimates
        return means, MLCs, modes
    
    
    def get_metrics(self, X_ten, y_ten, verbose=True):
        
        # Check for errors [REMOVE]
        for name, param in self.model.named_parameters():
            if param.requires_grad and torch.any(torch.isnan(param)):
                print("ERROR: nan parameters in model parameter {}. Check LR.".format(name))
                return -1
            
        X = X_ten.detach().cpu().numpy()
        y = y_ten.detach().cpu().numpy()
        
        metric_dict = {}
        
        # If invalid data, return empty metric dictionary
        if len(y) == 0:
            return metric_dict
        
        # Make prediction
        with torch.no_grad():
            mu, sigma, pi = self.model(X_ten)
            
        # Compute NLL
        nll_loss = self.gaussian_mixture_NLL(y_ten, mu, sigma, pi)
        metric_dict["nll"] = nll_loss.detach().cpu().numpy()
        
        # Compute interval metrics
        for p in [0.7, 0.9, 0.95, 0.99]:
            interval_width, interval_coverage = self.find_intervals(y_ten, mu, sigma, pi, p)
            metric_dict["int_width_{}".format(p)] = interval_width
            metric_dict["int_cover_{}".format(p)] = interval_coverage
        
        # Compute other metrics
        preds_mean, preds_MLC, preds_mode = self.find_dist_estimates(mu, sigma, pi)
        preds_mean = preds_mean * self.y_std + self.y_mean
        preds_MLC = preds_MLC * self.y_std + self.y_mean
        preds_mode = preds_mode * self.y_std + self.y_mean
        y = y * self.y_std + self.y_mean
        
        metric_dict["mse_mean"] = metrics.mean_squared_error(y, preds_mean)
        metric_dict["mse_MLC"] = metrics.mean_squared_error(y, preds_MLC)
        metric_dict["mse_mode"] = metrics.mean_squared_error(y, preds_mode)
        metric_dict["r2_mean"] = metrics.r2_score(y, preds_mean)
        metric_dict["r2_MLC"] = metrics.r2_score(y, preds_MLC)
        metric_dict["r2_mode"] = metrics.r2_score(y, preds_mode)
        
        if verbose:
            print("Performance - " + "".join("{}: {:.6f}, ".format(key, metric_dict[key]) for key in metric_dict))
            
        return metric_dict
    
    # Computes the p confidence intervals of our predicted distributions, and estimates how reliable these are
    def find_intervals(self, y_ten, mu, sigma, pi, p=0.95, n_bins=2001):
        
        # Compute pdf
        gmm_bins = self.get_bins(n_bins)
        gmm = self.gaussian_mixture_pdf(mu, sigma, pi, gmm_bins)
        gmm = gmm / torch.sum(gmm, axis=1, keepdim=True)
        
        # Compute cdf
        gmm = torch.cumsum(gmm, axis=1)
        
        # Find min and max probabilities
        p_min = (1 - p)/2
        p_max = 1 - p_min
            
        # Find interval of distribution within this range        
        interval_mins = torch.count_nonzero(gmm < p_min, axis=1)
        interval_maxs = n_bins - 1 - torch.count_nonzero(gmm > p_max, axis=1)
        interval_mins = gmm_bins[interval_mins]
        interval_maxs = gmm_bins[interval_maxs]
        interval_width = torch.mean(interval_maxs - interval_mins).item() * self.y_std
        
        # Find % of points that lie within these intervals
        preds_in_interval = (y_ten.flatten() > interval_mins) & (y_ten.flatten() < interval_maxs)
        interval_coverage = (torch.count_nonzero(preds_in_interval) / len(y_ten)).item()
        
        self.preds_in_interval = preds_in_interval
        
        return interval_width, interval_coverage
        
                
    # Computes negative log loss of gaussian mixture     
    def gaussian_mixture_NLL(self, y, mu, sigma, pi, return_nlls=False):
        mu = mu.view(-1, self.num_gaussians, 1)
        sigma = sigma.view(-1, self.num_gaussians, 1)
        comp = D.Independent(D.Normal(mu, sigma), 1)
        mix = D.Categorical(pi)
        gmm = D.MixtureSameFamily(mix, comp)
        nll = -gmm.log_prob(y)
        
        # Replace likelihoods that were smaller than e^-20 with e^-20 to avoid nans
        nll[nll > 20] = 20 
        
        nll_loss = torch.mean(nll)
        if return_nlls:
            return (nll_loss, nll)
        else:
            return nll_loss
        
        
    # Computes WLR loss 
    def WLR_loss(self, mu, sigma, pi, X_ten=None, inds_ten=None):
                
        # Determine whether to work with X_ten, or with inds_ten + previously saved training matrices
        if X_ten is None:
            if inds_ten is None:
                assert False, "Need either X or inds to compute WLR."
            else:
                reuse_train_matrices = True
                X_ten = self.X_lu_ten
                inds = inds_ten.detach().cpu().numpy().flatten()
        else:
            reuse_train_matrices = False
            
        # Useful variables
        batch_size = pi.shape[0]
        
        # Create bins to discretise PDFs
        bins, bins_stepsize = self.get_bins(return_stepsize=True)
        
        # Compute variables for kNN if they haven't been computed yet
        if not (reuse_train_matrices and hasattr(self, "WLR_vars")):
            
            # Compute distance matrix
            X = X_ten.detach().cpu().numpy()  
            D = kneighbors_graph(X, self.k, mode='distance', p=self.d)
            D = np.power(D.toarray(), self.d)
            D = np.maximum(D, D.T)
            
            # Kernel matrix
            K = np.exp(-D / (2*self.kernel_sigma**self.d))
            
            # Save variables for next time, to avoid re-computing
            self.WLR_vars = (D, K)
      
        # Reuse old KNN values if we've computed them
        else:
            D, K = self.WLR_vars
        
        # Subset to inds of this batch
        if not self.single_ssl_batch:
            D = D[np.ix_(inds, inds)]
            K = K[np.ix_(inds, inds)]
        
        # Find kNN pairs
        N_i, N_j = np.where(D > 0)
        pairs_keep = N_i > N_j # Remove half of Ws to avoid computing twice
        N_i = N_i[pairs_keep]
        N_j = N_j[pairs_keep]
        weights = np.asarray(K[N_i, N_j]).flatten()
        num_pairs = len(N_i)

        # Final weights
        weights = torch.tensor(K[N_i, N_j], dtype=torch.float32).flatten()

        # Mixture Gaussian PDF
        gmm_pdf = self.gaussian_mixture_pdf(mu, sigma, pi, bins)
        
        # Normalising PDF
        gmm_pdf = gmm_pdf * bins_stepsize
            
        # Compute difference in disitributions to compute 'local' Wasserstein
        diff_dist = gmm_pdf[N_i, :] - gmm_pdf[N_j, :]
        WLR_local = torch.cumsum(diff_dist, axis=1)
        
        # Find total WLR for each KNN pair
        WLR_dists = torch.sum(torch.abs(WLR_local.clone()), dim=1)
        WLR_dists = WLR_dists * bins_stepsize
        WLR_dists = torch.pow(WLR_dists, self.q)
        
        # Calculate and normalise WLR loss
        WLR_loss = torch.dot(weights, WLR_dists)
        WLR_loss = WLR_loss / num_pairs         
        WLR_loss = WLR_loss * self.ssl_weight
        
        return WLR_loss
    
    
    # Returns [num_mixtures x num_bins] array of 
    def gaussian_mixture_pdf(self, mu, sigma, pi, bins):
        n_mixtures = mu.shape[0]
        n_gaussians = mu.shape[1]
        n_bins = len(bins)
        gmm_pdf = torch.zeros([n_mixtures, n_bins])
        
        for k in range(n_gaussians):
            gmm_pdf = gmm_pdf + pi[:, k].view(-1, 1) * self.gaussian_pdf(mu[:, k].view(-1, 1), sigma[:, k].view(-1, 1), bins)
        return gmm_pdf
    
    
    # Returns 2D Gaussian [num_distributions x num_bins]
    def gaussian_pdf(self, mu, sigma, bins):
        sqrt_2_pi = 2.5066283
        val = 1 / (sigma * sqrt_2_pi) * torch.exp(-torch.pow(bins - mu, 2) / (2 * sigma * sigma))
        return val
    
    
    # Finds the median of smallest distance between points in our training set
    def get_sigma(self):
        if not self.is_ssl:
            return 0
        X = np.concatenate([self.X_l, self.X_u], axis=0)
        D = kneighbors_graph(X, n_neighbors=1, mode="distance", p=self.d).toarray()
        sigma = np.median(D[D>0]) * self.sigma_coef
        
        print("SSL Kernel Sigma has been set to {:.4f}".format(sigma))
        return sigma
    
    
    # Get bins to compute output distributions
    def get_bins(self, n_bins=None, return_stepsize=False):
        if n_bins is None:
            n_bins = self.b
        min_val = self.y_min - 0.1*(self.y_max - self.y_min)
        max_val = self.y_max + 0.1*(self.y_max - self.y_min)
        if return_stepsize:
            return torch.linspace(min_val, max_val, n_bins), (max_val - min_val) / (n_bins - 1)
        else:
            return torch.linspace(min_val, max_val, n_bins)
        
    