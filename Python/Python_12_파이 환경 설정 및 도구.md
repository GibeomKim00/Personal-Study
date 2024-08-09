# 파이 환경 설정 및 도구
### IDE(Integrated Development Environment)
: 텍스트 에디터, 디버거, 라이브러리 검색 등의 도구를 지원하는 GUI
* IDLE: 파이썬 표준 배포판에 포함되어 있는 IDE
* PyCharm
* IPython

### 코드 테스트
* pylint, pyflakes는 파이썬 코드 checker로 가장 인기 있는 모듈이다.
```
$pip install pylint
$pip install pyflakes

# 작성한 코드 파일을 평가해준다.
$ pylint style1.py
```

* unittest: 표준 라이브러리에서 제공하는 테스트 패키지로 로직을 테스트한다.
```
# cap.py
def just_do_it(test):
    from string import capwords
    return capwords(test)

# text_cap.py
import unittest
import cap

class Test(unittest.TestCase):
    # setUp 메서드는 각 테스트 메서드 전에 호출된다.
    def setUp(self):
        pass

    # tearDown 메서드는 각 테스트 메서드 후에 호출된다.
    def tearDown(self):
        pass

    def test_one_word(self):
        text = 'duck'
        result = cap.just_do_it(text)
        self.assertEqual(result, 'Duck;)
    
    def test_multiple_words(self):
        text = 'a vertiable flock of ducks'
        result = cap.just_do_it(text)
        self.assertEqual(result, 'A Vertiable Flock Of Ducks')

if __name__ == '__main__':
    unittest.main()
```

* doctest: 표준 라이브러리의 두 번째 테스트 패키지로 doctest를 사용하여 docstring 내에서 코멘트와 함께 테스트 코드를 작성할 수 있다.
```
def just_do_it(text):
    """
    >>> just_do_it('duck')
    'Duck'
    >>> just_do_it('a veritable flock of ducks')
    'A Veritable Flock Of Ducks'
    >>> just_do_it("I'm fresh out of ideas")
    "I'm Fresh Out Of Ideas"
    """
    from string import capwords
    return capwords(text)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
```

* nose: 함수 이름에 test가 포함되어 있다면, 테스트가 실행된다.
```
$ pip install nose

import cap
from nose.tools import eq_

def test_one_word():
    text = 'duck'
    result = cap.just_do_it(text)
    eq_(result, 'Duck')

def test_multiple_words(self):
    text = 'a vertiable flock of ducks'
    result = cap.just_do_it(text)
    eq_(result, 'A Vertiable Flock Of Ducks')
```

### 코드 디버깅
1. 파이썬에서 디버깅하는 가장 간다한 방법은 문자열을 출력하는 것이다.
    * vars(): 함수의 인자값과 지역 변수값을 추출한다.
```
def func(*args, **kwargs):
    print(vars())

>>> func(1,2,3)
출력값: {'args':(1,2,3), 'kwargs':{}}
```

2. 데커레이터(decorator)를 사용하면 함수 내의 코드를 수정하지 않고 함수 이전 혹은 이후에 코드를 호출할 수 있다.
```
def dump(func):
    def wrapped(*args, **kwargs):
        print("Function name: %s" % func.__name__)
        print("Input arguments: %s" % ' '.join(map(str,args)))
        print("Input keyword arguments: %s" % kwargs.items())
        output = func(*args, **kwargs)
        print("Output:", output)
        return output
    return wrapped

@dump
def double(*args, **kwargs):
    output_list = [2 * arg for arg in args]
    output_dict = {k:2*v for k, v in kwargs.items()}
    return output_list, output_dict

if __name__ == '__main__':
    output = double(3, 5, first=100, next=98.6, last=-40)

# 출력값
Function name: double
Input arguments: 3 5
Input keyword arguments: dict_item([('first', 100), ('next': 98.6), ('last', -40)])
Output: ([6, 10], {'first' : 200, 'next' : 197.2, 'last' : -80})
```

3. pdf: 표준 파이썬 디버거이다.
    * c(continue): 프로그램을 끝까지 실행
    * s(step): 파이썬 라인을 한 단계 앞으로 나아간다.
    * n(next): 한 단계 씩 나아가지만 함수 안으로 들어가지는 않는다.
    * l(list): 몇 줄의 코드를 볼 수 있다.
    * l #: 시작 라인 #부터 코드를 볼 수 있다
    * b #: # 번째 줄에 중단점 생성
    * b: 모든 중단점을 볼 수 있다. 

### 에러 메시지 로깅
* 로그(log): 일반적으로 메시지가 축적되는 시스템 파일이다.
* logging: 표준 파이썬 라이브러리 모듈이다.
```
import logging
logging.debug("Looks like rain")
logging.info("And hail")
logging.warn("Did I hear thunder?")
logging.error("Was that lightening?")
logging.critical("Stop fencing and get inside!")
```

### 코드 최적화
1. 시간 측정
    * time 모듈의 time 함수를 사용하여 프로그램 시작 전과 프로그램이 끝난 후의 시간 차를 구해서 측정한다.
    ```
    from time import time

    t1 = time()
    num = 5
    num *= 2
    print(time() - t1)
    ```

    * 표준 timeit 모듈을 사용하면 더 편하게 시간 측정이 가능하다.
    ```
    # timeit.timeit(code, number, count)
    from timeit import timeit
    print(timeit('num = 5; num *= 2', number=1))
    ```