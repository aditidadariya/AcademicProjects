#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:47:57 2022

@author: aditidadariya
"""

"""  Read_Files.py   """

"""####################################### Importing necessary libraries ########################################"""

import threading
from threading import *
from time import sleep
from Config import *
from Function_Library import *

"""#####################################  Calling functions to setup the data ###################################"""

# Reading the files using multi threading [1]
if __name__ == "__main__":
    # Create thread object for reading the Top30_airports_LatLong.csv file [2]
    thread1 = threading.Thread(name = 'ReadFileLines', target = ReadFileLines, args = (location_of_airportfile,'airport',))
    # Create thread object for reading the AComp_Passenger_data_no_error.csv file [2]
    thread2 = threading.Thread(name = 'ReadFileLines', target = ReadFileLines, args = (location_of_passengerfile, 'passenger',))
    # Start the thread 
    thread1.start()
    sleep(1)
    thread2.start()
    sleep(1)
    # Join the thread
    thread1.join()
    thread2.join()   
    
    # Verify if the above threads are still running
    if thread1.is_alive() or thread2.is_alive():
        sleep(0.1)
    else:
        # if the threads have stopped running, then initiate a new thread to update the airport name in passenger list [2]
        thread3 = threading.Thread(name = 'UpdateAirportName', target = UpdateAirportName, args = (datasetairport,datasetpassenger, ))
        thread3.start()         # Start the thread
        thread3.join()          # Join the thread


# Refereces:
# [1] https://www.geeksforgeeks.org/multithreading-python-set-1/
# [2] https://www.onooks.com/python-threading-error-group-argument-must-be-none-for-now/