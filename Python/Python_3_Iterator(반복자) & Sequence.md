# Iterator(반복자) & Sequence
### while문
* 중단하기: break
```
# break문이 수행되었는지 확인하는 법
cheeses = []
for cheese in cheeses: # 빈 리스트는 false에 해당 -> 반복문 수행 X
    print(cheese)
    break
else:   # 반복문이 동작하지 않았다면 break문도 동작하지 않았기에 else구문 동작
    print("Nothing happen")
```
### for문
```
rabbits = ['Flopsy','Mopsy','Cottontail','Peter']
current = 0
for current < len(rabbits):
    print(rabbits[current])
    current += 1

# 더 좋은 코드
# 리스트, 문자열, 튜플, 딕셔너리, 셋은 순회 가능한 객체이다.
for rabbit in rabbits:
    print(rabbit)

words = 'cat'
for word in words:
    print(word) # 출력값: c\n a\n t\n

accusation = {'room':'ballroom', 'weapon':'lead pipe', 'person':'Col. Mustard'}
for key in accusation:
    print(key) # 출력값: room\n weapon\n person\n
for value in accusation.values():
    print(value) # 출력값: ballroom\n lead pipe\n Col. Mustard
for item in accusation.items():
    print(item) # 출력값: ('room','ballroom')\n ('weapon','lead pipe')\n ('person':'Col. Mustard')\n
for key, value in accusation.items():
    print(key, value)
```

# 여러 sequence 순회하기: zip()
* 여러 sequence를 병렬로 순회하는 것
```
days = ['Monday','Tuesday',Wednesday']
fruits = ['banana', 'orange']
drinks = ['coffee','tea']
for day, fruit, drink in zip(days, fruits, drinks):
    print(day, fruit, drink)
# 출력값
Monday banana coffee
Tuesday orange tea
# 짧은 sequence가 완료되면 반복문은 종료된다.
```

# 숫자 sequence 생성하기: range()
* 형태: range(start, stop, step)
* start(기본값은 0)부터 stop - 1까지 진행
```
for x in range(0,3):
    print(x)

list(range(0,3))
```

# Comprehension(함축)
* 하나 이상의 iterator를 이용해 자료구조를 만드는 방법
* 리스트 comprehension
```
number_list = [number for number in range(1,6)]

a_list = [number for number in range(1,6) if number % 2 == 1]

# comprehension을 이용한 이중반복문
rows = range(1,4)
cols = range(1,3)
cells = [(row,col) for row in rows for col in cols] # 리스트 안에 튜플 존재
```

* 딕셔너리 comprehension
```
word = 'letters'
letter_counts = {letter: word.count(letter) for letter in set(word)}
# set을 통해 종복된 철자는 제외시킨다.(시간 단축)
```

* 셋 comprehension
```
a_set = {number for number in range(1,6) if number % 3 == 1}
```

# Generator
* 파이썬의 sequence를 생성하는 객체이다.
    * Ex) range()
* 한 번 iterator를 이용해 사용하면 리스트, 딕셔너리, 셋, 문자열과 같이 재사용할 수 없다.(메모리 절약)
* generator 함수: return 문을 통해 값을 반환하지 않고, yield 문으로 값을 반환한다.
```
def my_range(first=0, last=10, step=1):
    number = first
    while number < last:
        yield number
        number += step

ranger = my_range(1,5)
for x in ranger: # for문 이후로 ranger 재사용 불가
    print(x)
```

# None VS False
* None은 bool로 평가될 때 False처럼 보이지만 부울값의 False와 다르다.
```
def is_none(thing):
    if thing is None:
        print("It's None")
    elif thing:
        print("It's True")
    else:
        print("It's False)

is_none(None) # 출력값: It's None
is_none(True) # 출력값: It's True
is_none(False) # 출력값: It's False
```
