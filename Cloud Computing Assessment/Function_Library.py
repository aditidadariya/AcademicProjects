#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:54:24 2022

@author: aditidadariya
"""

""" Function_Library.py   """

"""####################################### Importing necessary libraries ########################################"""

import pandas as pd
from Config import *

"""######################################### Function declarations ##############################################"""

#===================================================================================================================
# Function to read  all the lines in the csv file
def ReadFileLines(location_of_file, file_keyword):
    
    # Open the csv file from its location and store it in file variable
    file = open(location_of_file, 'r')
    # Read all the lines in the csv file and store it in contentoffile variable
    content_of_file = file.readlines()

    # Iterate each line in the contentoffile variable and form a dictionary object variable
    for each_line in content_of_file:
        # Split eachline with ',' and store it in line variable
        line = each_line.strip().split(',')
        # Verify if line is not empty
        if len(line) > 1: 
             # Call the ConvertLinesDic function, which creates a dictionary(key,value) type of variable for each line
             dataset = ConvertLinesDic(line, file_keyword)  
    
    # Close the file         
    file.close()
    #Return the dataset established
    return dataset

#===================================================================================================================
# Function to convert the list of lists into dictionary object to store the dataset in a dictionary object form
def ConvertLinesDic(line, file_keyword):
    
    # Check if the file_keyword is 'airport', then creates a list of its elements
    if file_keyword == 'airport':
        # Assign each element of line to separate variable
        airport_name, airport_code, lat, long = line
        # Form a list with airport_name and airport_code
        listoflines_airport = airport_name,airport_code
        # Append the a new list with the line formed above
        datasetairport.append(listoflines_airport)
        # Return the airport list
        return datasetairport
        
    # Check if the file_keyword is 'passenger', then creates a list of its elements
    if file_keyword == 'passenger':
        # Assign each element of line to separate variable
        pasngr_id, flight_id, from_airport, dest_airport, dep_time, flight_time = line
        # Form a list with airport_name and airport_code
        listoflines_pasngr = pasngr_id,flight_id,from_airport
        # Append the a new list with the line formed above
        datasetpassenger.append(listoflines_pasngr)
        # Return the passenger list
        return datasetpassenger

#===================================================================================================================
# Function to update the ariport_code to airport_name in passengerdataset
def UpdateAirportName(airportdataset, passengerdataset):
    print(type(airportdataset),type(passengerdataset))
    # Iterate each line of the passengerdataset to compare and update the airport_code to airport_name
    for pdata in passengerdataset:
        # Assign each element of line to separate variable
        pasngr_id,flight_id,from_airport = pdata
        # Verify if line is not empty
        
        if len(airportdataset)>0:
            # Iterate each line of the airportdataset to compare and update the airport_code to airport_name
            
            for adata  in airportdataset:
                # Assign each element of line to separate variable
                airport_name,airport_code = adata
                # Verify if from_airport from passengerdataset is equal to airport_code of airportdataset
                
                if airport_code == from_airport:
                    # Update the list of passengers to have airport_name
                    updatedlistofpasngr = pasngr_id,flight_id,airport_name
                    # Print the updatedlistofpasngr
                    print(updatedlistofpasngr)
                    # Append the updatedlistofpasngr into datasetupdatedpassenger to store all the lists together
                    datasetupdatedpassenger.append(updatedlistofpasngr)
                    break


