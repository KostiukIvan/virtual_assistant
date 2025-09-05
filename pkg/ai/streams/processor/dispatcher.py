import threading


# It's communication states between caller and receiver:
# Caller agenda:
# -----     SILENCE
# ******    LISTENING
# ^^^^^^    SPEAKING
# ---s---   SHORT PAUSE
# ---L---   LONG PAUSE
#
# Receiver agenda:
# ******    LISTENING
# ^^^^^^    SPEAKING
#
# Time flow: ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Caller:    -----------------------------------^^^^^^^^^^^^^^^^^^^--------s------^^^^^^^^^^^^^^^^^^^^^^----------s-------------L******************************^^^^^^^^^^^^^^
# Receiver:  ********************************************************************************************************************^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^**************
# 
# C SILENCE     -> R LISTENING
# C SPEAKING    -> R LISTENING
# C LISTENING   -> R SPEAKING
# C SHORT PAUSE -> R LISTENING
# C LONG PAUSE  -> R SPEAKING 
# 
# R LISTENING   -> C SPEAKING, C SILENCE, C SHORT PAUSE, 
# R SPEAKING    -> C LISTENING 
class Caller:
    def __init__(self):
        self.
        self.silence = threading.Event()
        self.speaking = threading.Event()
        self.short_pause_detected = threading.Event()
        self.long_pause_detected = threading.Event()

    # SILENCE
    def is_silence(self):
        return self.silence.is_set()
    
    def set_silence_mode(self):
        self.silence.set()
        print("[[LISTENING]]")

    # SPEAKING
    def is_speaking(self):
        return self.speaking.is_set()
    
    def set_speaking_mode(self):
        self.speaking.set()
        print("[[SPEAKING]]")

    # LONG PAUSE 
    def is_long_pause_detected(self):
        return self.long_pause_detected.is_set()

    def set_long_pause_detected(self):
        self.long_pause_detected.set()
        
    # SHORT PAUSE
    def is_long_pause_detected(self):
        return self.short_pause_detected.is_set()

    def set_long_pause_detected(self):
        self.short_pause_detected.set()



class Receiver:
    def __init__(self):
        self.speaking = threading.Event()
        self.listening = threading.Event()
        
    # SPEAKING
    def is_speaking(self):
        return self.speaking.is_set()
    
    def set_speaking_mode(self):
        self.speaking.set()
        self.listening.clear()
        
    # LISTENING
    def is_listening(self):
        return self.listening.is_set()
    
    def set_listening_mode(self):
        self.listening.set()
        self.speaking.clear()