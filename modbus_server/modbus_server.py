#!/bin/python

from pyModbusTCP.server import ModbusServer, DataBank
from time import sleep
from random import uniform


  
# Driver Code
if __name__=="__main__":

    # create a modbus server instance
    server = ModbusServer("localhost",no_block=True)

    try:
        print("Starting server...")
        server.start()
        print("Server is online")
        state = [0]
        while True:
            continue
            # DataBank.set_words(0, [int(uniform(0,100)),5,6,8])
            # if state != DataBank.get_words(1):
            #     state = DataBank.get_words(1)
            #     print("Value of Register 1 has changed to " + str(state))
            #     sleep(0.5)
    except KeyboardInterrupt:
        print("Server shutting down...")
        server.stop()
        print("Server is offline")
