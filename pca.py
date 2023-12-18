import numpy as np
import pandas as pd
from eig_utils import *
from typing import Literal

class PCA():
    def __init__(self, n_components=None, solver: Literal["auto", "QR", "Jacobian"]="auto"):
        self.n_components = n_components
        self.__covariance_matrix__ = None
        self.__eigenvalues__ = None
        self.__eigenvectors__ = None
        self.explained_variance = None
        self.__standardized_data__ = None
        self.components_ = None
        self.__solver__ = solver
        self.mean_ = None

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def fit(self, data):
        if self.n_components is None:
            self.n_components = min(*data.shape)
        
        self.mean_ = data.mean(axis=0)
        standardized_data = (data - self.mean_)
        self.__covariance_matrix__ = np.cov(standardized_data, rowvar = False)
        
        eigvalues, eigvectors = None, None
        if self.__solver__ == "auto":
            eigvalues, eigvectors = np.linalg.eig(self.__covariance_matrix__)
        elif self.__solver__ == "QR":
            eigvalues, eigvectors = qr_method(self.__covariance_matrix__)
            eigvectors = -eigvectors
        elif self.__solver__ == "Jacobian":
            eigvalues, eigvectors = rotation_method(self.__covariance_matrix__)
    
        self.__eigenvalues__, self.__eigenvectors__ = eigvalues, eigvectors

        order_of_importance = np.argsort(self.__eigenvalues__)[::-1]

        sorted_eigenvalues = self.__eigenvalues__[order_of_importance]
        self.__sorted_eigenvectors__ = -self.__eigenvectors__[:,order_of_importance]

        self.components_ = (self.__sorted_eigenvectors__.T)[:self.n_components,:]
        self.explained_variance_ratio_ = (sorted_eigenvalues / np.sum(sorted_eigenvalues))[:self.n_components]

    def transform(self, data):
        if self.__eigenvectors__ is None:
            raise "Model doesn't fitted. Use method fit"
        
        standardized_data = (data - data.mean(axis=0))
        
        try:
            return np.dot(standardized_data, self.__sorted_eigenvectors__[:,:self.n_components])
        except:
            raise  "Something went wrong. Hint: dimensions can be wrong size of the fitted"

    def get_covariance(self):
        if self.__covariance_matrix__ is None:
            raise "Cov matrix doesn't exists. First, fit the model. Use method fit"
        
        return self.__covariance_matrix__
