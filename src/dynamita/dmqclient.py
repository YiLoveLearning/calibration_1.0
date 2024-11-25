from ctypes import cdll, c_int, c_char_p, CFUNCTYPE
import socket
import random
import platform
import sys
import os
import threading
import time
from datetime import datetime
from . import tool as dtool

if platform.system() == 'Windows':  
    import winreg

class BackgroundExecutable(threading.Thread):
    def __init__(self, parent, command, name='background-executable-thread'):
        self.command = command
        self.parent = parent
        super(BackgroundExecutable, self).__init__(name=name)
        self.start()

    def run(self):
        os.system(self.command)
        self.parent.running = False

class DMQProcess:
    def __init__(self, sumo_path, license_file : str):
        self.running = True
        dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        cmd = f"{sumo_path}/DMQ.exe -autoexit -run \"{license_file}\" >dmq{dt}.log"
        g = BackgroundExecutable(self, cmd)
    
        
class DMQClient:
    pass

class DMQDllWrapper:
    def __init__(self):
        self.version = 'Sumo22'
        self.platform_name = '' 
        self.sumo_path = ""
        self.license_file = ""
        self._load_sumo()
    
    def _load_sumo(self):
        library_prefix = ''
        library_ext = ''
        self.platform_name = platform.system()
        if self.platform_name == 'Windows':
            library_ext = 'dll'
            library_prefix = ''
            
                        
            self.sumo_path = self.get_install_location()                       
            
            if self.sumo_path == "":
                raise FileNotFoundError(self.version + ' is not installed')
                
            self.license_file = self.get_license_location()
            if self.license_file == "":
                raise FileNotFoundError(self.version + ' has no license')
                
            DMQProcess(self.sumo_path, self.license_file)
                
            if sys.version_info[0] > 3 or (sys.version_info[0] == 3 and sys.version_info[1] >= 8):
                os.add_dll_directory(self.sumo_path) # Python 3.8
        elif self.platform_name == 'Linux':
            library_ext = 'so'
            library_prefix = 'lib'
        elif self.platform_name == 'Darwin':
            library_ext = 'dylib'
            library_prefix = 'lib'
        else:
            raise NotImplementedError('Unsupported platform: '+self.platform_name)

        dmqClient_filename = os.path.join(
            self.sumo_path, library_prefix + "DMQClient." + library_ext
        )
        if os.path.isfile(dmqClient_filename):
            cwd = os.getcwd()
            os.chdir(self.sumo_path)
            # load DMQClient.dll from install
            self.dmq_dll = cdll.LoadLibrary(dmqClient_filename)
            os.chdir(cwd)
        else:
            raise FileNotFoundError('DMQClient file not found: ' + dmqClient_filename)
        
        self.dmq_dll.initModule.argtypes = [c_char_p]         
        self.dmq_dll.createQueue.argtypes = []   
        self.dmq_dll.createQueue.restype = c_char_p        
        self.dmq_dll.createSpecQueue.argtypes = [c_char_p]
        self.dmq_dll.createSpecQueue.restype = c_char_p
        self.dmq_dll.openQueue.argtypes = [c_char_p] 
        self.dmq_dll.openQueue.restype = c_char_p
        self.dmq_dll.sendText.argtypes = [c_char_p, c_char_p] 
        self.dmq_dll.sendText.restype = c_int
        self.dmq_dll.getText.argtypes = [c_char_p, c_int]                
        self.dmq_dll.getText.restype = c_char_p
        self.dmq_dll.closeQueue.argtypes = [c_char_p] 
        self.dmq_dll.getVersion.restype = c_int
        self.dmq_dll.applyLicense.argtypes = [c_char_p]
        self.dmq_dll.applyLicense.restype = c_char_p

        # call initModule("Python") from the dll
        self.dmq_dll.initModule("Python".encode("utf8"))
    
    if platform.system() == 'Windows':    
        def get_license_location(self):
            aKey = "SOFTWARE\\Dynamita\\" + self.version +"\\PATHS"
            aReg = winreg.ConnectRegistry(None,winreg.HKEY_CURRENT_USER)
            try:
                aKey = winreg.OpenKey(aReg, aKey)
                val = winreg.QueryValueEx(aKey, "License")
                return val[0]
            except:
                return ""
                
        def get_install_location(self):
            aKey = "SOFTWARE\\Dynamita\\" + self.version +"\\PATHS"
            aReg = winreg.ConnectRegistry(None,winreg.HKEY_CURRENT_USER)
            try:
                aKey = winreg.OpenKey(aReg, aKey)
                val = winreg.QueryValueEx(aKey, "INST")
                return val[0]
            except:
                aKey = "SOFTWARE\\Dynamita\\" + self.version +"\\PATHS"
                aReg = winreg.ConnectRegistry(None,winreg.HKEY_LOCAL_MACHINE)
                try:
                    aKey = winreg.OpenKey(aReg, aKey)
                    val = winreg.QueryValueEx(aKey, "INST")
                    return val[0]
                except:
                    return ""

    
    def create(self):
        dmqc = DMQClient()
        dmqc.dmq_dll = self
        dmqc.key_p = self.dmq_dll.createQueue()
        dmqc.key = dmqc.key_p.decode('utf8')
        return dmqc
        
    def create_specific_queue(self, key: str) -> DMQClient:
        dmqc = DMQClient()
        dmqc.dmq_dll = self
        dmqc.key_p = self.dmq_dll.createSpecQueue(key.encode("utf8"))
        dmqc.key = dmqc.key_p.decode('utf8')
        return dmqc
        
    def open(self, key: str) -> DMQClient:
        dmqc = DMQClient()
        dmqc.dmq_dll = self
        dmqc.key_p = self.dmq_dll.openQueue(key.encode("utf8"))
        dmqc.key = dmqc.key_p.decode('utf8')
        return dmqc
           
    def apply_new_license(self):
        self.license_file = self.get_license_location()
        if self.license_file == "":
            raise FileNotFoundError(self.version + ' has no license')
        return self.dmq_dll.applyLicense(self.license_file.encode("utf8")).decode("utf8")

            
DMQ = DMQDllWrapper()

class DMQClient:
    def __init__(self):       
        self.key = "N/A"
        self.key_p = None
        
    def send_data(self, msg: str):
        return DMQ.dmq_dll.sendText(self.key_p, msg.encode("utf8"))
    
    def read_data(self, blocking: bool = False):
        blocking_int = 0
        if blocking:
            blocking_int = 1
        ptr = DMQ.dmq_dll.getText(self.key_p, blocking_int)
        result = ptr.decode("utf8")
        return result
        
    def close(self):
        DMQ.dmq_dll.closeQueue(self.key_p)
        self.key_p = None
        self.key = ""
           

class OPCClient:    
    def __init__(self, address: str, config: str):
        self.dmq = DMQ.create()
        self.lock = threading.Lock()
        self.running = True
        cmd = f"{DMQ.sumo_path}/OPCClient.exe \"{address}\" \"{config}\" \"{self.dmq.key}\""
        g = BackgroundExecutable(self, cmd)
        
    def read_variables(self, variables: list) -> dict:
        self.lock.acquire()
        self.dmq.send_data(f"read ,{','.join(variables)}")
        res = {}
        line = self.dmq.read_data(True)
        while line != "ReadEnd" and line != "CLOSED":
            args = line.split("|")
            if (len(args) >= 2):
                res[args[0]] = dtool.convert_to_data((args[1].split("="))[1])
            else:
                print("ERROR in read_variables response: {line}")
            line = self.dmq.read_data(True)    
        self.lock.release()
        return res
        
    def write_variables(self, variables: dict) -> dict:
        self.lock.acquire()
        a = []
        for key in variables:
            a.append(f"{key}={variables[key]}")
        self.dmq.send_data(f"write ,{','.join(a)}")
        res = {}
        line = self.dmq.read_data(True)
        while line != "WriteEnd" and line != "CLOSED":
            args = line.split("|")
            if (len(args) >= 2):
                res[args[0]] = dtool.convert_to_data((args[1].split("="))[1])
            else:
                print("ERROR in write_variables response: {line}")
            line = self.dmq.read_data(True)  
        self.lock.release()
        return res
    
    def read_mapped_variables(self, mapped_variables: list, condition):
        lst = []
        result = {}
        for x in mapped_variables:
            if condition(x):
                lst.append(x.opc_tag)
        data = self.read_variables(lst)
        for x in mapped_variables:
            if condition(x):
                if x.sumo_type == "REAL":
                    result[x.sumo_name] = data[x.opc_tag] / x.scaling - x.offset
                elif x.sumo_type == "INT":
                    result[x.sumo_name] = int(data[x.opc_tag] / x.scaling - x.offset)
                else:
                    result[x.sumo_name] = data[x.opc_tag]
        return result
                
    def close(self):
        self.lock.acquire()
        self.dmq.close()
        self.lock.release()

class SumoGUI:
    def __init__(self, project: str):
        self.model_initialized = False
        self.project_loaded = False
        self.dmq = DMQ.create()
        self.running = True
        self.last_message = ""
        self.project_path = ""
        cmd = f"{DMQ.sumo_path}/{DMQ.version}.exe \"{project}\" -dmq \"{self.dmq.key}\""
        g = BackgroundExecutable(self, cmd)
        
    def communicate(self):
        line = self.dmq.read_data(False)
        
        if (line == "CLOSED"):
            self.last_message = "CLOSED"
            return False
            
        if (line != ""):
            print(f"SumoGUI: {line}")
            self.last_message = line
        else:
            return False
        if line.startswith("project_init "):
            self.project_path = line.replace("project_init ", "")
        elif (line == "project_loaded"):
            self.project_loaded = True
        elif (line == "model_init"):
            self.model_initialized = True
        elif (line == "model_unloaded"):
            self.model_initialized = False
            
        return True
        
    def wait_for(self, l):
        while not l():
            if not self.communicate():
                time.sleep(1)   

    def set_variable(self, variable : str, value):
        self.dmq.send_data(f"core_cmd set {variable} {value};")
        
    def set_variables(self, data):
        s = ""
        for y in data:
            x = data[y]
            s = f'{s}core_cmd set {y} {x};\n'
        self.dmq.send_data(s)
    
    def onstart(self, command : str):
        self.dmq.send_data(f"onstart {command};")
        
    def push_button(self, tab : str, button : str, param : str = ""):
        if (param == ""):
            self.dmq.send_data(f"button {tab} {button}")
        else:
            self.dmq.send_data(f"button {tab} {button} {param}")
        
    def select_maintab(self, tab : str):
        self.dmq.send_data(f"maintab {tab}")
        
    def core_command(self, command : str):
        self.dmq.send_data(f"core_cmd {command};")
                
    def close(self):
        self.dmq.close()