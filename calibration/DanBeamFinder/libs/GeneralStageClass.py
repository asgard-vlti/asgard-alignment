import zmq
import time
import numpy as np
class GeneralStageObject:
    def __init__(self,host="192.168.100.2",port=5555):
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.context.socket(zmq.REQ)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 2000)
        server_address = f"tcp://{self.host}:{self.port}"
        self.socket.connect(server_address)
        self.state_dict = {"message_history": [], "socket": self.socket}
        
# def send_and_get_response(message):
#     # st.write(f"Sending message to server: {message}")
#     state_dict["message_history"].append(
#         f":blue[Sending message to server: ] {message}\n"
#     )
#     state_dict["socket"].send_string(message)
#     response = state_dict["socket"].recv_string()
#     if "NACK" in response or "not connected" in response:
#         colour = "red"
#     else:
#         colour = "green"
#     # st.markdown(f":{colour}[Received response from server: ] {response}")
#     state_dict["message_history"].append(
#         f":{colour}[Received response from server: ] {response}\n"
#     )

    # return response.strip()

    def Get_pos(self,stage:str,beam:int):
        message = f"read {stage}{beam}"
        # response = send_and_get_response(message)
        self.state_dict["socket"].send_string(message)
        response = self.state_dict["socket"].recv_string()
        return float(response)


    # this is a bit of a complicated funciton but it is meant to ensure the set_pos holds the code up and waits till the
    # postion is actually reached. 
    def Set_pos(self,stage:str,beam:int,pos:float,tol:float=1e-1,timeout:float=10,poll_dt:float=0.01,settle_polls: int=3):
        # these three lines are what actually move the the stage
        message = f"moveabs {stage}{beam} {pos}"
        self.state_dict["socket"].send_string(message)
        response = self.state_dict["socket"].recv_string()

        # this is the while loop that will check if the stage has reached its location
        start = time.perf_counter()
        good_count = 0
        last_pos = None
        while True:
            current_pos = self.Get_pos(stage,beam)
            # print(current_pos)
            last_pos = current_pos

            if abs(current_pos - pos)<= tol:
                good_count += 1
                if good_count >= settle_polls:
                    return response
            else:
                good_count = 0

            if time.perf_counter()- start > timeout: 
                raise TimeoutError(f"Stage {stage}{beam} did not reach target.\n Target={pos}, lase positions ={last_pos},Tolerance={tol}")
            time.sleep(poll_dt)
    
    @staticmethod
    def rasterScanSnakePattern(StartX,StartY,StepAwayFromStartX,StepAwayFromStartY,StepCountX, StepCountY,
                                XminLimit=0,XmaxLimit=10000,YminLimit=0,YmaxLimit=10000):
        xmax=StartX + StepAwayFromStartX
        xmin=StartX - StepAwayFromStartX
        if xmin<XminLimit:
            xmin=XminLimit
            print("You have gone beyond the min X limit of the mount which is default set to 0 you can change this by passing a new limit value into this function ")
        if xmax>XmaxLimit:
            print("You have gone beyond the Max X limit of the mount which is default set to 0 you can change this by passing a new limit value into this function ")
            xmax=XmaxLimit

        ymax=StartY + StepAwayFromStartY
        ymin=StartY - StepAwayFromStartY
        if ymin<YminLimit:
            ymin=YminLimit
            print("You have gone beyond the min X limit of the mount which is default set to 0 you can change this by passing a new limit value into this function ")
        if ymax>YmaxLimit:
            print("You have gone beyond the Max X limit of the mount which is default set to 0 you can change this by passing a new limit value into this function ")
            ymax=YmaxLimit

        x=np.linspace(xmin,xmax,StepCountX)
        y=np.linspace(ymin,ymax,StepCountY)

        gridpoints = np.zeros((StepCountX,StepCountY,2))
        
        for iy in range(StepCountY):
            for ix in range(StepCountX):
                if iy % 2==0:
                    gridpoints[iy,ix,0]= x[ix]
                else:
                    gridpoints[iy,ix,0]= x[StepCountX-1-ix]
                gridpoints[iy,ix,1]= y[iy]
        return gridpoints

