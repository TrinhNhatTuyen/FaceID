import threading
from firebase import firebase
class PutAlertThread(threading.Thread):
    def __init__(self, homeid, value):
        threading.Thread.__init__(self, name = "Put-Alert")
        self.homeid = homeid
        self.value = value

    def run(self):
        self.put_alert()

    def put_alert(self):
        homefirebase = firebase.FirebaseApplication('https://vina-ai-8ce99-default-rtdb.firebaseio.com', None)
        Location = "Muong_Thanh"
        homefirebase.put(Location, self.homeid, self.value)
