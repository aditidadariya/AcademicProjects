#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 22:18:49 2022

@author: aditidadariya
"""

""" Reducer_2.py """

"""####################################### Importing necessary libraries ########################################"""

import sys
import re
from Config import *

"""############################################ Function declaration ############################################"""

# Reducer_2 function is defined to get the passenger having had maximum number of flights
def Reducer_2():
    
    # prev_pscode, prev_pscount variables are defined to hold the initial null value
    prev_pscode = None
    prev_pscount = 0
    
    # Iterate each line of the line received from Shuffle_Sort.py 
    for line in sys.stdin:
        # Split the line
        line = line.strip().split('\t', 1)
        # Assign each element of line to separate variable
        code, count = line
        # Remove the special characters from code
        code1 = re.sub("[()]","", code)
        # Remove the special characters from code1
        newcode = re.sub("'","",code1)
        # Remove the special characters from count
        newcount = re.sub("[()]","",count)
        
        # Convert the string count into int while handling the exceptions
        try:
            count = int(newcount)
        except ValueError:
            continue
        
        # Split the newcode by the '_'
        newpscode, newflcode = newcode.strip().split('_', 1)
        # For a list with three elements
        pflist = newpscode, newflcode, count
        # Form a list of lists created above to store all the keys of passengers and thier values
        passengerflighttotal.append(pflist)
        

    # Iterate to each list of passengerflighttotal to get a passenger with total number of flights
    for eachlist in passengerflighttotal:
        # Assign count to 0
        count = 0
        # Assign each element of eachlist to separate variable
        passenger_code, flight_code, prev_pscount = eachlist
        # Verify if the passenger_code is different from prev_pscode
        if prev_pscode != passenger_code:
            # Iterate each list of passengerflighttotal to count the total number of flights
            for each in passengerflighttotal:
                # Assign each element of each to separate variable
                passenger_code1, flight_code, prev_pscount = each
                # Verify if the passenger ids match, then increament the count by 1 and for pflist
                if passenger_code == passenger_code1:
                    count += 1
                else:
                    continue
            # Replace prev_pscode with the passenger_code
            prev_pscode = passenger_code
            # Form pflist to store the total count of flight for each passenger
            pflist = passenger_code, count
            # Append the passengerflightreduced with list of all passengers with thier total number of flights
            passengerflightreduced.append(pflist)
    
    # Defined maxflightspasgnr variable to store the max key value pair
    maxflightspasgnr = {}
    # Iterate passengerflightreduced to set the max default
    for each in passengerflightreduced:
        maxflightspasgnr.setdefault(each[1], []).append(each)
    # Get the max flight value
    max_flights = max(maxflightspasgnr)
    # Print the max Key, value pair
    print(maxflightspasgnr[max_flights])
    
    # Save the output in txt file
    with open('Task2_PassengerMaxFlights.txt', 'a') as f:
        f.seek(0)           # Set the point to begining in txt file
        f.truncate()        # Erase all the content in txt file
        f.write(str(maxflightspasgnr[max_flights])) # write the output
        f.close()
        
        
"""############################################## Function Call ###############################################"""

# Call Reducer_2 function to get the total number of passengers having had maximum number of flights
Reducer_2()


# References: 
# https://stackoverflow.com/questions/64812189/find-max-value-in-a-nested-list