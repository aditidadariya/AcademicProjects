#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 20:57:05 2022

@author: aditidadariya
"""

""" Shuffle_Sort.py """

"""####################################### Importing necessary libraries ########################################"""

import sys
from Config import *

"""########################################## Function declaration ##############################################"""

# Shuffle_Sort function is defined to sort the imput of key and value getting from mappers
def Shuffle_Sort():
    # Iterate to all the inputs coming from mapper
    for lines in sys.stdin:
        # Split the line by tab
        lines = lines.strip().split('\t', 1)     
        # Input line is split into Key and Value
        code, count = lines
        # Key are taken into a list
        sortlist.append(code)                           
   
    # Keys are sorted
    sortlist.sort()
    # Iterate over the list of keys are print the Key and Value pair
    for each in sortlist:
        print('%s\t%s' % (each, 1))
           
"""############################################## Function Call ###############################################"""
    
# Call Shuffle_Sort function to Sort the Key and Values     
Shuffle_Sort()

