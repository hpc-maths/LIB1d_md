# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:19:59 2022

@author: ali.asad
"""
import numpy as np
import json

from src.lib.post_process import time_series as y_ts

# json to numpy arrays
#%%
class NumpyEncoder(json.JSONEncoder):
    """ Recursively converts numpy array into lists for JSON serialization """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def JSONtoNumpy(dico, key_hist=''):
  """ Recursively converts all list of floats (and list of list of floats) to numpy arrays in the input dictionnary """
  print(key_hist)
  if isinstance(dico, dict):
    for key in dico.keys():
      dico[key] = JSONtoNumpy( dico[key], key_hist='.'.join( (key_hist, key) ) )
  else:
    if isinstance(dico, list):
      if len(dico)>0:
        if isinstance(dico[0], list) or isinstance(dico[0],dict):
          for i in range(len(dico)):
            dico[i] = JSONtoNumpy(dico[i], key_hist='.'.join( (key_hist, str(i))))
        elif isinstance(dico[0], np.ndarray):
          dico = np.array(dico)
        elif isinstance(dico[0], float):
          return np.array(dico)
        elif isinstance(dico[0], int):
          return np.array(dico)
  return dico
#%%


# function to write dictionary to json
#%%
def dict_to_json(dictionary, filename):
    
    if filename[:-5] not in ['.json']:
        filename = filename + '.json'
    
    with open(filename, "w") as outfile:
        json.dump(dictionary, outfile)
    outfile.close()
#%%

# function defining dictionary for 1D solution state
#%%
def write_solution_state(solution, options_1, options_2, filename, transform=True, verbose=False):
    x = (np.r_[options_1['mesh']['cellX'], options_2['activematerial']['mesh']['cellX'], options_2['currentcollector']['mesh']['cellX']])
    
    dictionary = {}
    dictionary["x"] = x.tolist()
    dictionary["t"] = solution.t.tolist()
    dictionary["y"] = solution.y.tolist()
    
    dict_to_json(dictionary, filename)
    
    if verbose:
        print(f"solution state file written with name {filename}")
#%%

# function defining dictionary for 1D solution in dimensional form
#%%
def write_solution(solution, options_1, options_2, filename, transform=True, verbose=False):
    if transform:
        x, time, aux_a, c_e, phi_e, aux_c, c_s, phi_s = y_ts.get_y_t(solution.t, solution.y, options_1, options_2)
    dictionary = {}
    dictionary["x"] = x.tolist()
    dictionary["t"] = time.tolist()
    dictionary["aux_a"] = aux_a.tolist()
    dictionary["ce"] = c_e.tolist()
    dictionary["phie"] = phi_e.tolist()
    dictionary["aux_c"] = aux_c.tolist()
    dictionary["cs"] = c_s.tolist()
    dictionary["phis"] = phi_s.tolist()
    
    dict_to_json(dictionary, filename)
    
    if verbose:
        print(f"solution file with dimensions written with name {filename}")
#%%

# function to read solution
#%%
def read_solution(filename):
    solution = []
    with open(filename, "r") as infile:
        dictionary = json.load(infile)
    
    for key in dictionary:
        solution.append( np.array(dictionary[key]) )
  
    return solution