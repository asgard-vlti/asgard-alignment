import zmq
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

    def Set_pos(self,stage:str,beam:int,pos:float):
        message = f"moveabs {stage}{beam} {pos}"
        self.state_dict["socket"].send_string(message)
        response = self.state_dict["socket"].recv_string()
        return response