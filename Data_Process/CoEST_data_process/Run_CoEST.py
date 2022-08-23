from common import main as common
from EasyClinic import main as ec
from HIPAA import main as hi
from IceBreaker import main as ib
from InfusionPump import main as ip
from Kiosk import main as k
from WARC import main as w

if __name__ == '__main__':
    datasets_dir = r'..\..\data\CoEST'
    common(datasets_dir)
    ec(datasets_dir)
    hi(datasets_dir)
    ib(datasets_dir)
    ip(datasets_dir)
    k(datasets_dir)
    w(datasets_dir)
