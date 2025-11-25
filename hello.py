def add_and_double(a, b):
	# 在这行设置断点以开始调试（点击行号左侧的空白处）
	c = a + b
	d = c * 2
	return d

def main():
    # 定义变量a并赋值为8
	a = 8 #  定义变量a并赋值为8
    # 定义变量b并赋值为9
	b = 9
    # 调用add_and_double函数，将a和b作为参数传入，并将返回值存储在result变量中
	result = add_and_double(a, b)
    # 打印result变量的值
	print('result', result)


if __name__ == '__main__':
	main()
