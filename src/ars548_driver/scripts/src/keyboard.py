#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from pynput import keyboard

def on_press(key):
    try:
        # 发布的消息内容是按键的字符表示
        msg = String()
        msg.data = f'Key pressed: {key.char}'
        pub.publish(msg)
    except AttributeError:
        # 如果按键没有字符（如特殊键），可以处理其他情况
        msg = String()
        msg.data = f'Special key pressed: {key}'
        pub.publish(msg)

def on_release(key):
    # 这里可以处理按键释放事件，若需要的话
    if key == keyboard.Key.esc:
        # 结束监听器
        return False

if __name__ == '__main__':
    rospy.init_node('keyboard_listener')

    # 创建一个 Publisher
    pub = rospy.Publisher('/keyboard_topic', String, queue_size=1)

    # 使用 pynput 来监听键盘事件
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        rospy.loginfo("Listening for keyboard input...")
        rospy.spin()
        listener.join()
