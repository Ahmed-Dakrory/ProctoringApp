from winreg import *
import psutil

REG_PATH = r"SOFTWARE\Proctoring"

def set_reg(PATH,name, value):
    try:
        CreateKey(HKEY_CURRENT_USER, PATH)
        registry_key = OpenKey(HKEY_CURRENT_USER, PATH, 0, 
                                       KEY_WRITE)
        SetValueEx(registry_key, name, 0, REG_SZ, value)
        CloseKey(registry_key)
        return True
    except WindowsError as e:
        print(e)
        return False


def get_reg(PATH,name):
    try:
        registry_key = OpenKey(HKEY_CURRENT_USER, PATH, 0,
                                       KEY_READ)
        value, regtype = QueryValueEx(registry_key, name)
        CloseKey(registry_key)
        return value
    except WindowsError:
        return None


# set_reg('GUID', str('asko98dsjsdjk'))
set_reg(r"Software\\Classes\\Proctoring\\",'URL Protocol', '')
set_reg(r"Software\\Classes\\Proctoring\\",'', 'URL:Alert Protocol')




# set_reg(r"Software\\Classes\\Proctoring\\Shell\\Open\\command",'', '\"D:\\AnyDesk.exe\" \"%1\"')
# print(get_reg('GUID'))

PROCNAME = "Taskmgr.exe"

for proc in psutil.process_iter():
    # check whether the process name matches
    print(proc.name())
    if proc.name() == PROCNAME:
        proc.kill()