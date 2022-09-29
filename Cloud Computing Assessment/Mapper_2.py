#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 17:23:39 2022

@author: aditidadariya
"""

""" Mapper_2.py """

"""####################################### Importing necessary libraries ########################################"""

import sys
from Config import *

"""############################################ Function declaration ############################################"""

# Mapper_2 function is defined to map the passenger_id and flight_id keys
def Mapper_2():
    # Iterate each line of input line
    for line in sys.stdin:
        # Split the line by comma
        singleline = line.strip().split(',')
        # Verify if the line is empty
        if len(singleline) > 1:
            # Remove all the special characters of 1st element of passenger_id in the line
            singleline[0] = ''.join(e for e in singleline[0] if e.isalnum())
            # Remove all the special characters of 2nd element of flight_id in the line
            singleline[1] = ''.join(e for e in singleline[1] if e.isalnum())
            # Combine passenger_id and flight_id
            passengerflight = singleline[0]+"_"+ singleline[1]
            # Map each passengerflight key to 1 and print it            
            print('%s\t%s' % (passengerflight, 1))
            # Append the flightairportlist to store all the lists created in flightairport
            passengerflightlist.append(passengerflight)   
    # Return passengerflightlist        
    return passengerflightlist

"""############################################## Function Call ###############################################"""

# Call Mapper_2, which maps the Key passengerflight with Value 1    
Mapper_2 ()


