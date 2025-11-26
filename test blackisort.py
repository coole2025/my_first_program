import os, sys
import numpy


def test():
    """
    这是一个测试函数

    该函数目前不执行任何操作，仅作为示例使用
    """
    print("This is a test function.")
    pass  # pass语句在这里用作占位符，表示函数体为空


def my_func(x, y):
    if x > 0:
        return x + y
    else:
        return x - y


def add(x: int, y: int) -> int:
    return x + y


add(5, 3)  # 这行应该出现黄色波浪线警告

if __name__ == "__main__":
    test()
    result = my_func(5, 3)
    print(f"Result: {result}")
