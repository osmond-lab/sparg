#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:56:05 2023

@author: puneeth
"""

import numpy as np
import sympy as sym

# def find_infinite_solutions(A, b):




def MLE(S_inv, loc, rootind, n) :  
    S_inv = np.array(S_inv)
    lenloc = len(loc)
    # one_vect = np.ones((lenloc,1)) 
    mu_vect = np.zeros((lenloc,1)) 
    roots = np.unique(rootind)
    k = len(roots)
    D  = np.zeros((k,lenloc))
    for i,root in enumerate(roots): 
        D[i][np.where( np.array(rootind) == root)[0]] += 1.0
    
    A = np.matmul(D,np.matmul(S_inv,np.transpose(D))) #Matrix of coefficients of the system of linear equations 
    b = np.matmul(D,np.matmul(S_inv,loc)) #Vector of constants of the system of linear equations. 
    augmented_matrix = np.column_stack((A, b)) # Construct the augmented matrix [A|b]
    rre_form, pivots = sym.Matrix(augmented_matrix).rref() # Perform row reduction on the augmented matrix
    null_space = sym.Matrix(A).nullspace() # Find the basis of the null space of A i.e. all x such that Ax = 0
    
    if len(pivots) == A.shape[0]:
        if int(A.shape[0]) in pivots: 
            raise TypeError("No Solutions")
        else: 
            mu_list = np.array(rre_form.col(-1))
            mu_vect = np.matmul(np.transpose(D),mu_list)    
            sigma = np.matmul(np.matmul(np.transpose(loc - mu_vect), S_inv), (loc - mu_vect))/n
            
    else: 
        mu_particular = rre_form.col(-1)
        sigma_list = []
        
        for rep in range(20): 
            mu_list = mu_particular
            for col in null_space:
                a = np.random.uniform(-50,50)
                mu_list += a*col
            mu_vect = np.matmul(np.transpose(D),mu_list)    
            sigma = np.matmul(np.matmul(np.transpose(loc - mu_vect), S_inv), (loc - mu_vect))/n
            sigma_list += [sigma]
    
    
    
    return mu_list, sigma
    
    
# Example usage
# A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = np.array([1, 2, 3])

A = np.array([[1, 1], [2, 2]])
b = np.array([1,3])






    
 

# Check if the system has infinite solutions
# if num_pivots < A.shape[1]:
#     print("The system has infinite solutions.")
    
#     # Find the particular solution
#     particular_solution = np.linalg.lstsq(A, b, rcond=None)[0]
#     print("Particular solution:", particular_solution)
    
#     # Find the basis for the null space
#     null_space_basis = np.linalg.qr(A.T, mode='r')[1][:, num_pivots:]
#     print("Null space basis:", null_space_basis)
# else:
#     print("The system does not have infinite solutions.")

# find_infinite_solutions(A, b)
