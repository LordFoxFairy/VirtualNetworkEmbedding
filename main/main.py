from util import config
from data_loader import predata
from algorithms.model import NetModel as LModel

# 测试函数-目的用于测试数据
# test.test_func()

def read_data():
    """
    solution:
    SN_Link:(s1,s2,sbandwidth)
    SN_Node:(snode)
    VN_Link:(v1,v2,vbandwidth)
    VN_Node:(vnode)
    VN_Life:{时序key:[[序号num,生命周期period,开始时间start_time,结束时间end_time],]}
    :return:
    """
    (solution, SN_Link, SN_Node, VN_Link, VN_Node, VN_Life) = predata.read_SN_VN(config.SnFile, config.VnFile)
    return (solution, SN_Link, SN_Node, VN_Link, VN_Node, VN_Life)

def run():
    # 读取数据e
    (solution, SN_Link, SN_Node, VN_Link, VN_Node, VN_Life) = predata.read_SN_VN(config.SnFile, config.VnFile)

    model = LModel(solution, SN_Link, SN_Node, VN_Link, VN_Node, VN_Life)

    data = {
        "SN_Link": SN_Link,
        "SN_Node": SN_Node,
        "VN_Node": VN_Node,
        "VN_Link": VN_Link,
        "VN_Life": VN_Life,
        "solution": solution
    }
    model.experience(data)

# run()