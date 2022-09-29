#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:01:30 2022

@author: aditidadariya
"""

""" Mapper_1.py """

"""####################################### Importing necessary libraries ########################################"""

import sys
from Config import *

"""############################################ Function declaration ############################################"""

# Defined Mapper_1 function to map the flight_id and airport_name keys
def Mapper_1():
    
    # Iterate each line of input line
    for line in sys.stdin:
        # Split the line by comma
        singleline = line.strip().split(',')
        # Verify if the line is empty
        if len(singleline) > 1:
            # Remove all the special characters of 3rd element of airport_name in the line
            singleline[2] = ''.join(e for e in singleline[2] if e.isalnum())
            # Remove all the special characters of 2nd element of flight_if in the line
            singleline[1] = ''.join(e for e in singleline[1] if e.isalnum())
            # Combine flight_id with airport_name
            flightairport = singleline[2]+"_"+ singleline[1]
            # Map each flightairport key to 1 and print it
            print('%s\t%s' % (flightairport, 1))
            # Append the flightairportlist to store all the lists created in flightairport
            flightairportlist.append(flightairport)       
    # Return flightairportlist
    return flightairportlist
        
"""############################################## Function Call ###############################################"""

# Call Mapper_1, which maps the Key flightairport with Value 1
Mapper_1()

