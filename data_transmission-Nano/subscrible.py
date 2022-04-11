#!/usr/bin/env python
import rospy
from std_msgs.msg import String, ByteMultiArray

import struct
import numpy as np



def callback(data):
    receive_dic = {}
    bbox_list = []

    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    result = np.array(data.data, dtype = np.uint8)

    receive_dic["time_stamp"] = struct.unpack("q", bytes(result[:8]))[0]
    receive_dic["is_iframe"] = struct.unpack("?", bytes(result[8:9]))[0]
    receive_dic["bbox_size"] = struct.unpack("q", bytes(result[9:17]))[0]

    for i in range(struct.unpack("q", bytes(result[9:17]))[0]):
        base = 17 + 24 * i
        # label||bbox.x||bbox.y||bbox.weight||bbox.height||bbox.box_id
        _tmp_list = struct.unpack("iffffi", bytes(result[base:base+24]))
        bbox_list.append(_tmp_list)
    receive_dic["bbox"] = bbox_list    

    print("-----")
    print(receive_dic)
    print("-----")


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('apple', anonymous=True)

    rospy.Subscriber("bbox_publish", ByteMultiArray, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
