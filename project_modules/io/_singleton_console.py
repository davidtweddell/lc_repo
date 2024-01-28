from rich.console import Console

class SingletonConsole:
    _instance = None

    @staticmethod
    def getInstance(*args, **kwargs):
        if SingletonConsole._instance == None:
            SingletonConsole(*args, **kwargs)
        return SingletonConsole._instance

    def __init__(self, *args, **kwargs):
        if SingletonConsole._instance != None:
            raise Exception("This class is a singleton!")
        else:
            SingletonConsole._instance = Console(*args, **kwargs)

