def func1():
    print('func1')

def func2():
    print('func2')

def func3():
    print('func3')
number = 1

func_map = {1:func1, 2:func2}

# func_map.get(number, func3)()


numDic = [7, 6,1, 2, 3, 4, 5]

#
# fruits  = ['a', 'b', 'c', 'd']
# for index, fruites in enumerate(fruits):
#     print(f"index:{index} fruites:{fruites}")

# print(any(x % 2 == 0 for x in numDic))
# print(all(x % 2 == 0 for x in numDic))

# sq = lambda x: x ** 2
# print(sq(2))


print(list(filter(lambda x:x % 2 == 0, numDic)))
print(list(map(lambda x:x % 2 == 0, numDic)))
print(sorted(numDic))

#print(lambda x: x**2, numDic)

numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
print(list(set(numbers)))

from collections import Counter
print(Counter(numbers))


class A:
    def method(self):
        print("Method is from class A")

class B:
    def method(self):
        print("Method is from class B")

class C(B,A):
    pass
instance  = C()
instance.method()


def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("something is happen before is called")
        result = func(*args, **kwargs)
        print("something happen after is called")
        return result
    return wrapper

# def my_decorator(func):
#     def wrapper(*args, **kwargs):
#         print("Something is happening before the function is called.")
#         result = func(*args, **kwargs)
#         print("Something is happening after the function is called.")
#         return result
#     return wrapper

# @my_decorator
# def saySHR():
#     print('Heloooo shahram')
#
# saySHR()


# def my_decorator(func):
#     def wrapper(*args, **kwargs):
#         print("Something is happening before the function is called.")
#         result = func(*args, **kwargs)
#         print("Something is happening after the function is called.")
#         return result
#     return wrapper
#
# @my_decorator
# def say_hello():
#     print("Hello!")
#
# say_hello()


# x=10
# def my_fun():
#     y=20
#     x=20
#     print(f'x:{x}')
#     #print(locals())
#     print(globals()['x'])
#
# my_fun()
#
#
#
# from collections import  deque
# queue = deque()
# queue.append(1)
# queue.append(2)
# queue.append(3)
# queue.append(4)
# print(queue.popleft())
# print(queue.pop())
# print(queue.pop())
#
#
# import random
# numbers = [1, 2, 3, 4, 5]
# random.shuffle(numbers)
# print(numbers)

# my_list = [1, 2, 3]
# attributes = dir(my_list)
# #print(attributes)
#
# import sys
# if len(sys.argv) > 1:
#     file_name = sys.argv[1]
#     print(f"File name: {file_name}")
#
#
# name = ['a', 'b', 'c']
# age = [10, 100, 1000]
# for name, age in zip(name, age):
#     print(f"{name} is {age} years old")

people = [
{'name': 'Alice', 'age': 30},
{'name': 'Bob', 'age': 55},
{'name': 'Charlie', 'age': 35}]

oldperson = max(people, key=lambda x:x['age'])
print(oldperson)

yanperson = min(people, key=lambda x:x['age'])
print(yanperson)



import datetime
# Get Current Date and Time
current_datetime = datetime.datetime.now()
# Format Date
formatted_date = current_datetime.strftime('%Y-%m-%d')
# Calculate Time Difference
date1 = datetime.datetime(2023, 1, 1)
date2 = datetime.datetime(2023, 12, 31)
time_difference = current_datetime - date1

#print(time_difference)


numbers = [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
sublist = numbers[2:5] # Output: [2, 3, 4]
print(sublist)

li1 = [1,2,3]
li2 = [1,2,3]

if li1 is li2:
    print('same')
else:
    print('diff obj')

if li1 == li2:
    print('same vale')
else:
    print('diff value')

import timeit
def myfun():
    return sum(range(100000))

start_time = timeit.default_timer()
myfun()
print(timeit.default_timer() - start_time)


# exec_time = timeit.timeit(myfun(), number=100)
#
# print(f"exec time:{exec_time} second")

import inspect
def my_function(a, b=10, *args, c=20, **kwargs):
    pass

# Get Function Arguments
argspec = inspect.getfullargspec(my_function)
source_code = inspect.getsource(my_function)
print(source_code)


def anyFunc(x):
    assert x>0, "x must be positive"


numbers = [1, 2, 3, 4, 5]
even_numbers = list(filter(lambda x: x % 2 == 0, range(1,30)))
print(even_numbers)


class Meta(type):
    def __new__(cls, name, bases, dct):
        print(f"Creating class {name}")
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=Meta):
    pass

x  = MyClass()

def add(a: int, b : int) -> int:
    return a * b

print(f"add in {add(5,5)}")


from abc import  ABC, abstractmethod

class Terminal(ABC):
    name = ''
    @abstractmethod
    def my_abc_method(self):
       self.name = 'Terminal'

class POS(Terminal):
    def my_abc_method(self):
        print(f"class:{self.__class__.__base__}")

class IPG(Terminal):
    def my_abc_method(self):
        print(f"class:{self.__class__.__base__}")

class ATM(Terminal):
    def my_abc_method(self):
        print(f"class:{self.__class__.__base__}")

mobj = POS()
mobj .my_abc_method()
print(f"class name :{mobj.name}")

if isinstance(mobj, POS):
    print(f"class2:{mobj.__class__.__base__}")


class myiter:
    def __iter__(self):
        self.num = 1
        return self

    def __next__(self):
        self.num *= 2
        return self.num


def mygen():
    num = 1
    while True:
        num *= 2
        yield num

iter_obj = myiter()
gen_obj = mygen()

print(next(iter(iter_obj)))
print(next(gen_obj))

#*******************************************************thread
import asyncio

async  def mycoroutine():
    print('start')
    await asyncio.sleep(1)
    print('end')

asyncio.run(mycoroutine())

import aiofiles
async def read_file():
    async with aiofiles.open('file.txt', mode='r') as file:
            contents = await file.read()
            print(contents)


asyncio.run(read_file())


from concurrent.futures import ThreadPoolExecutor

def task(n):
    return n * 2

with ThreadPoolExecutor() as executor:
    results = executor.map(task, [1, 2, 3])
    print(list(results))



# def square(n):
#     return n * n
#
# with ThreadPoolExecutor() as executor:
#     results = executor.submit(square, 3)
#     print(results.result())
#
# from multiprocessing import Pool
# def square_number(n):
#     return n * n
#
# numbers = [1, 2, 3, 4]
# with Pool() as pool:
#     results = pool.map(square_number, numbers)
#     print(results) # Output: [1, 4, 9, 16]

from multiprocessing import Process
# def print_numbers():
#     for i in range(10):
#         print(i)

# process1 = Process(target=print_numbers)
# process2 = Process(target=print_numbers)
# process1.start()
# process2.start()
# process1.join()
# process2.join()
#*****************************************************************************************************


# from pyspark import SparkContext
#
# sc = SparkContext("local", "MLTrain")
# numbers = sc.parallelize([1,2,3,4])
# squares = numbers.map(lambda x:x*x)
# print(squares.collect())


# import cProfile
#
# def factorial(n):
#     return 1 if n == 0 else n * factorial(n-1)
# def main():
#     print(factorial(10))
#
# cProfile.run('main()')


#The functools.lru_cache d e c o r a t o r c a c h e s t h e r e s u l t s o f f u n c t i o n c a l l s ,
#
from functools import lru_cache
@lru_cache
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

import ast
source_code = 'a = 5 + 2'
parsed_code = ast.parse(source_code)
print(ast.dump(parsed_code))

def chain(*iterables):
    for iterable in iterables:
        yield from iterable


result = list(chain([6,5],[4,3],[2,1]))
print(result)


from unittest.mock import MagicMock
mock=MagicMock(return_value=fibonacci(10))
mock2=MagicMock(return_value=list(chain([6,5],[4,3],[2,1])))
print(mock())


import threading

def print_number():
    for i in range(5):
        print(i)


thread = threading.Thread(target=print_number())
thread.start()
thread.join()


data = bytearray(b"hi, mr shahram rostami")
view = memoryview(data)
chunk = view[8:13]
print(bytes(chunk))


import inspect

def my_func():
    pass

print(inspect.isfunction(my_func))







