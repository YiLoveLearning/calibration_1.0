from ctypes import cdll, c_int, c_char_p, CFUNCTYPE
import os
import platform
import sys


class _Duration:
    """Data structure to hold duration constants."""
    pass

dur = _Duration()

dur.msec = 1
dur.sec  = 1000 * dur.msec
dur.min  = 60 * dur.sec
dur.hour = 60 * dur.min
dur.day  = 24 * dur.hour
dur.week = 7 * dur.day


# Generates an array from a flexible (allowing floats) range specification
# The user need to specify, begin and two of end, step and count.
def create_array(begin, end = None, step = None, count = None, includeEnd = True):
    missing = 0
    if (step is None):
        missing += 1
    if (count is None):
        missing += 1
    if (end is None):
        missing += 1
    if (missing < 1):
        raise Exception("Overdefined array. Define one less argument.") 
    elif (missing > 1):
        raise Exception("Underdefined array. Define two arguments from end, count and step")
        
    if (step is None):               
        if includeEnd:
            step = (end - begin)/(count - 1)
        else:
            step = (end - begin)/(count)
        return [begin + x * step for x in range(count)]
    elif (count is None):
        count = int((end - begin + step * 0.99)/step)
        if (count < 0):
            step = -step
            count = int((end - begin + step * 0.99)/step)
        result = [begin + x * step for x in range(count)]
        if includeEnd:
            result.append(end)
        return result
    elif (end is None):
        return [begin + x * step for x in range(count)]
    
    assert False, "create_array reached the end"



class SumoScheduler:
    def __init__(self, sumoPath=""):
        # Preparing some convenience member fields.
        self.version = 'Sumo22'
        self.platform_name = ''
        self.message_callback = None
        self.datacomm_callback = None
        self.scheduledJobs = 0
        self._load_sumo(sumoPath)
        self.jobData = { }
        self.persistent = "persistent"
        self.VERSION = "22.0.0"
        if (self.getDLLVersion() != self.getPYVersion()):
            raise Exception(f"Version mismatch: sumoscheduler.py is version {self.getPYVersion()} but sumoscheduler.dll is {self.getDLLVersion()}")


    def _load_sumo(self, sumoPath=""):
        library_prefix = ''
        library_ext = ''
        self.platform_name = platform.system()
        if self.platform_name == 'Windows':
            library_ext = 'dll'
            library_prefix = ''
            import winreg

            def getInstallLocation():
                aKey = "SOFTWARE\\Dynamita\\" + self.version +"\\PATHS"
                aReg = winreg.ConnectRegistry(None,winreg.HKEY_LOCAL_MACHINE)
                try:
                    aKey = winreg.OpenKey(aReg, aKey)
                    val = winreg.QueryValueEx(aKey, "INST")
                    return val[0]
                except:
                    aKey = "SOFTWARE\\Dynamita\\" + self.version +"\\PATHS"
                    aReg = winreg.ConnectRegistry(None,winreg.HKEY_CURRENT_USER)
                    try:
                        aKey = winreg.OpenKey(aReg, aKey)
                        val = winreg.QueryValueEx(aKey, "INST")
                        return val[0]
                    except:
                        return ""

            if sumoPath == "":
                sumoPath = getInstallLocation()
            if sumoPath == "":
                raise FileNotFoundError(self.version + 'is not installed')
            if sys.version_info[0] > 3 or (sys.version_info[0] == 3 and sys.version_info[1] >= 8):
                os.add_dll_directory(sumoPath) # Python 3.8
        elif self.platform_name == 'Linux':
            library_ext = 'so'
            library_prefix = 'lib'
        elif self.platform_name == 'Darwin':
            library_ext = 'dylib'
            library_prefix = 'lib'
        else:
            raise NotImplementedError('Unsupported platform.')

        core_filename = os.path.join(
            sumoPath, library_prefix + "sumoscheduler." + library_ext
        )
        if os.path.isfile(core_filename):
            cwd = os.getcwd()
            os.chdir(sumoPath)
            self.scheduler = cdll.LoadLibrary(core_filename)
            os.chdir(cwd)
        else:
            raise FileNotFoundError('Core file not found: ' + core_filename)

        self.scheduler.schedule.argtypes = [c_char_p, c_char_p, c_char_p, c_int]
        self.scheduler.schedule.restype = c_int
        self.scheduler.setParallelJobs.argtypes = [c_int]
        self.scheduler.setMaxJobReuse.argtypes = [c_int]
        self.scheduler.finish.argtypes = [c_int]
        self.scheduler.sendCommand.argtypes =[c_int, c_char_p]
        self.scheduler.setLogDetails.argtypes = [c_int]
        self.scheduler.getScheduledJobs.restype = c_int
        self.scheduler.getVersion.restype = c_int


        def convertToData(s):
            if (";" in s):
                result = []
                for item in s.split(";"):
                    result.append(convertToData(item))
                return result
            try:
                return float(s)
            except ValueError:
                return s


        def internal_datacomm_callback(job, msg):
            if self.datacomm_callback is not None:
                data = {}
                lst = msg.decode('utf8').split("|")
                for pair in lst:
                    args = pair.split(" = ")
                    data[args[0]] = convertToData(args[1])
                self.datacomm_callback(job, data)
            return 0


        def internal_message_callback(job, msg):
            if (msg is not None) and (self.message_callback is not None):
                self.message_callback(job, msg.decode('utf8'))
            return 0;


        CALLBACKFUNC = CFUNCTYPE(c_int, c_int, c_char_p)
        self.c_message_callback = CALLBACKFUNC(internal_message_callback)
        self.c_datacomm_callback = CALLBACKFUNC(internal_datacomm_callback)

        self.scheduler.register_message_callback.argtypes = [CALLBACKFUNC]
        self.scheduler.register_message_callback(self.c_message_callback)
        self.scheduler.register_datacomm_callback.argtypes = [CALLBACKFUNC]
        self.scheduler.register_datacomm_callback(self.c_datacomm_callback)


    def schedule (self, model, commands, variables, blockDatacomm=False, jobData=None):
        varstr = "|".join(variables)
        id = self.scheduler.schedule(model.encode("utf8"), (";".join(commands)).encode("utf8"), varstr.encode("utf8"), int(blockDatacomm))
        self.jobData[id] = jobData
        self.scheduledJobs = self.scheduler.getScheduledJobs()
        return id

    def setParallelJobs(self, jobs):
        self.scheduler.setParallelJobs(jobs)


    def setMaxJobReuse(self, reuse):
        self.scheduler.setMaxJobReuse(reuse)


    def finish(self, job):
        self.scheduler.finish(job)
        if (self.jobData[job] == None or not (self.persistent in self.jobData[job]) or  self.jobData[job][self.persistent] != True):
            del self.jobData[job]
        self.scheduledJobs = self.scheduler.getScheduledJobs()

    def deleteJobData(self, job):
        del self.jobData[job]

    def sendCommand(self, job, command):
        self.scheduler.sendCommand(job, command.encode("utf8"))


    def getJobData(self, jobId):
        return self.jobData[jobId]


    def isSimFinishedMsg(self, msg):
        return msg.startswith("530004")


    def setLogDetails(self, level):
        self.scheduler.setLogDetails(level)
        
    def getPYVersion(self):
        return self.VERSION
    
    def getDLLVersion(self):
        v = self.scheduler.getVersion()
        b = v % 1000
        v = int(v/1000)
        m = v % 1000
        v = int(v/1000)
        return f"{v}.{m}.{b}"

    def cleanup(self):
        self.jobData.clear()
        self.scheduler.cleanup()


sumo = SumoScheduler()
