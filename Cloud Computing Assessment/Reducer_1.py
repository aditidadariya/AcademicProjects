#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:18:15 2022

@author: aditidadariya
"""

""" Reducer_1.py """

"""####################################### Importing necessary libraries ########################################"""

import sys
import re
from Config import *

"""############################################ Function declaration ############################################"""

# Defined Reducer_1 function to give the total number of flight from each airport
def Reducer_1():
    
    # prev_code, prev_airportcode and prev_count variables are defined to hold the initial null value
    prev_code = None
    prev_airportcode = None
    prev_count = 0
    
    # Iterate each line of the line received from Shuffle_Sort.py 
    for line in sys.stdin:
        # Split the line
        line = line.strip().split('\t', 1)
        # Assign each element of line to separate variable
        code, count = line
        # Remove the special characters from code [2]
        code1 = re.sub("[()]","", code)
        # Remove the special characters from code1 [1]
        newcode = re.sub("'","",code1)
        # Remove the special characters from count [2]
        newcount = re.sub("[()]","",count)

        # Convert the string count into int while handling the exceptions
        try:
            count = int(newcount)
        except ValueError:
            continue
        
        # Compare if newcode read from the input line is equal to the previous code
        if prev_code == newcode:
            # If both the codes are same, then increament the previous count with count
            prev_count += count
        else:    
            # If both the codes are different then, 
            # assign count to prev_count
            prev_count = count
            # assign newcode to prev_code
            prev_code = newcode
            # Split the prev_code 
            prev_code1 = prev_code.strip().split('_', 1)
            # Assign each element of prev_code1 to separate variable
            airport_name, flight_code = prev_code1
            # Form falist to with airport_name, flight_code, prev_count
            falist = airport_name, flight_code, prev_count
            # Create a new list to store all the lists formed above
            flightairportreduced.append(falist)
        
    # Iterate to each list of flightairportreduced to get total number of flights from each airport
    for eachlist in flightairportreduced:
        # Assign count to 0
        count = 0
        # Assign each element of eachlist to separate variable
        airport_name, flight_code, prev_count = eachlist
        # Verify if the airport_code is different from prev_airportcode
        if prev_airportcode != airport_name:
            # Iterate each list of flightairportreduced to count the total number of flights
            for each in flightairportreduced:
                # Assign each element of each to separate variable
                airport_name1, flight_code, prev_count = each
                # Verify if the airport names match, then increament the count by 1 and replace prev_airportcode
                if airport_name == airport_name1:
                    count += 1
                    prev_airportcode = airport_name
                else:
                    continue
            # Print the Airport Name with its total number of flights
            print('%s - Total no of flights = %s' % (airport_name, count)) 
            # Save the output in txt file
            with open('Task1_FlightEachAirport.txt', 'a') as f:
                f.write(str('%s - Total no of flights = %s' % (airport_name, count)))   # write the output
                f.write('\n')       # Enter the next line
                f.close()           # Close the file
       

"""############################################## Function Call ###############################################"""

# Call Reducer function to get the total number of flights from each Airport
Reducer_1()
 
# Reference: 
# [1] https://stackoverflow.com/questions/5843518/remove-all-special-characters-punctuation-and-spaces-from-string
# [2] https://www.delftstack.com/howto/python/python-remove-parentheses-from-string/


