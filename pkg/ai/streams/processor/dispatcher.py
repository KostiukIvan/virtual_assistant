import threading


class Dispatcher:
    def __init__(self):
        self.listening = threading.Event()
        self.speaking = threading.Event()
        self.set_listening_mode()

    def is_listening(self):
        return self.listening.is_set()

    def is_speaking(self):
        return self.speaking.is_set()

    def set_listening_mode(self):
        self.listening.set()
        self.speaking.clear()
        print("[[LISTENING]]")

    def set_speaking_mode(self):
        self.listening.clear()
        self.speaking.set()
        print("[[SPEAKING]]")
