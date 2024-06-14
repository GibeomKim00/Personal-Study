# Function(함수)
### Positional arguments(위치 인자)
* arguments를 순서대로 parameter(매개변수)에 복사한다.
* 단점: 각 parameter의 의미를 알고 있어야 순서에 맞게 argument를 보낼 수 있다.
```
def menu(wine, entree, dessert):
    return {'wine':wine, 'entree':entree, 'dessert':dessert}

# 서로 다른 결과 값을 얻게 된다.
menu('chardonnay', 'chicken', 'cake')
menu('chicken', 'chardonnay', 'cake)
```

### 키워드 인자
* 함수의 parameter 순으로 맞춰서 argument를 보낼 필요가 없다.
* 위치 인자와 키워드 인자를 혼합해서 사용할 수 있으나 **반드시** 위치 인자가 키워드 인자보다 먼저 와야 한다.
```
# 같은 결과값을 얻게 된다.
menu(entree = 'chicken', dessert = 'cake', wine = 'chardonnay')
menu(wine = 'chardonnay', entree = 'chicken', dessert = 'cake')

# 위치 인자와 키워드 인자의 혼합
menu('chardonnay', dessert = 'cake', entree = 'chicken')
menu('chiken', wine = 'chardonnay', dessert = 'cake') # wine argument의 value가 2개 이상 받기에 오류 발생
```

### 기본 매개변수값 지정 시 주의사항
* 기본 인자값은 함수가 실행될 때 정의되지 않고, 함수를 **정의**할 때 계산된다.
```
def buggy(arg, result=[]):
    result.append(arg)
    print(result)

buggy('a') # 출력값: ['a']
buggy('b') # 출력값: ['a','b']

# 위 함수 수정
def buggy(arg):
    result = []
    result.append(arg)
    print(result)

buggy('a') # 출력값: ['a']
buggy('b') # 출력값: ['b']
```

### positional argument(위치 인자) 모으기: *
* 함수의 parameter에 asterisk(*) 사용 시 위치 인자를 튜플로 묶어준다.
```
def print_args(*args):
    print('Positional argument tuple:', args)

print_args() # 출력값: Positional argument tuple: ()
print_args(3, 2, 1, 'wait!') # 출력값: Positional argument tuple: (3,2,1,'wait!')
```
* unpacking: *를 통해 리스트, 튜플을 **를 통해 딕셔너리를 언패킹할 수 있다.
```
def add(a, b):
    return a + b

item_list = [1, 3]
result = add(*item_list)
```

### 키워드 인자 모으기: **
* 키워드 인자를 딕셔너리로 묶어준다.
```
def print_kwargs(**kwargs):
    print('Keyword arguments:', kwargs)

print_kwargs(wine = 'merlot', entree = 'mutton', dessert = 'macaroon')
# 출력값
Keyword arguments: {'wine':'merlot', 'entree':'mutton', 'dessert':'macarron'}
```
 
* *args와 **kwargs를 동시에 사용하기 위해서는 순서대로 배치해야 한다.

### 함수를 argument로 받는 함수
* python은 **'모든 것이 객체이다'**.
* 함수도 객체의 일부분이다.
```
def answer():
    print(42)

def run_something(func):
    func()


run_something(answer) # 함수를 객체 취급
# 출력값: 42

# argument가 여러 개인 함수인 경우 위치 인자 또는 키워드 인자 사용
def sum_args(*args):
    sum(args) # sum()은 파이썬 내장 함수이다.

def run_with_positional_args(func, *args):
    func(args)

run_with_posiotional_args(sum_args, 1, 2, 3, 4)
```
* **파이썬에서 괄호 ()는 함수 호출을 의미하고, 괄호가 없을 경우 함수를 객체처럼 취급하겠다는 의미이다.**

## closure
* 클로져는 다른 함수에 의해 <span style="color:yellow">**동적**</span>으로 생성되며, 외부 함수의 변수값을 알고 있는 함수이다.
```
def knights2(saying):
    def inner2():
        return "We are the knights who say : '%s'" % saying
    return inner2

a = knights2('Duck') # 변수 a의 type은 function이면서 closure이다.
a() # 출력값: We are the knights who say : 'Duck'
```
* 내부 함수(inner2)는 외부 함수의 argument를 사용하고 있다.
* 외부 함수는 내부 함수를 호출하지 않고 반환하고 있다. 이때, 내부 함수의 복사본이 반환된다.

## anonymous function(익명 함수): lambda function(람다 함수)
```
def fruit_name(fruits, func):
    for fruit in fruits:
        print(func(fruit))

def capital_fruit_name(fruit):
    return fruit.capitalize() + '!'

fruits = ['apple', 'banana', 'orange']
fruit_name(fruits, capital_fruit_name)

# lamda함수 이용
fruit_name(fruits, lambda fruit: fruit.capitalize() + '!')
```

## decorator
* 하나의 함수를 취해서 또 다른 함수를 반환하는 함수이다.
```
def document_it(func):
    def new_function(*args, **kwargs):
        print(args)
        print(kwargs)
        result = func(*args, **kwargs) # *args는 3 5가 **kwargs에는 아무것도 아무것도 !!!  없다.
        print(result)
        return result
    return new_function

def add_ints(a, b):
    return a + b

cooler_add_ints = document_it(add_ints) # decorator를 수동으로 할당
cooler_add_ints(3, 5)
# 출력값
(3, 5)
{}
8

# decorator 수동 할당 대신 사용 가능
# decorator를 쓰고 싶은 함수 위에 @decorator함수_이름 적기
@document_it
def add_ints(a,b):
    return a + b

add_ints(3,5)
# 출력값
(3, 5)
{}
8
```
* 함수는 여러 개의 decorator를 가질 수 있으며 함수에 가장 가까운(def 바로 위) decorator부터 실행한다.
```
def square_it(func):
    def new_function(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * result
    return new_function

@document_it
@square_it
def add_ints(a, b):
    return a + b:

add_ints(3,5)
# 출력값
(3, 5)
{}
64
```

## namespace & scope
* 함수로부터 전역 변수(global variable)의 값을 얻을 수 있지만 함수 안에서 전역 변수의 값을 바꿀 수는 없다.
```
number = 1
def print_number():
    print(number)

print_number() # 출력값: 1

def print_number():
    print(number)
    number = 7  # 오류 발생
    print(number)
```

```
number = 1
def print_number():
    number = 7
    print(number, id(number))

# 위의 값은 local namespace에 있는 number이고 아래 값은 global namespace에 해당하기에 값이 서로 다르다.
print_number()
id(number)
```
* global 키워드를 통해 함수 내의 지역 변수가 아닌 전역 변수를 접근할 수 있다.
* **python 철학**: 명확한 것이 함축적인 것보다 낫다.
```
number = 1
def change_number_print_global():
    global number
    number = 7 # 전역 변수의 값을 변경
    print(number)

change_number_print_global() # 출력값: 7
number # 출력값: 7
```
* namespace의 내용을 접근하기 위한 두 가지 함수
   * locals(): local namespace의 내용이 담긴 딕셔너리를 반환
   * globals(): global namespace의 내용이 담긴 딕셔너리 반환