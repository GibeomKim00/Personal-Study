# module & package & program
### module
* 파이썬 코드의 파일이다.
* import를 통해 다른 모듈의 코드를 사용할 수 있다.
* 모듈의 이름이 같은 경우 또는 모듈의 이름이 긴 경우 alias를 사용한다.
```
import matplotlib as plt
```
* 모듈 안에 필요한 함수만 import 하기(메모리 절약)
```
from report import get_description as dp_it
```

### package
* module을 모아 놓은 directory이다.
* directory에 __ init __.py 파일이 있어야 파이썬이 이 directory를 package로 간주한다.

### 표준 라이브러리
* Counter(): 항목을 세주는 함수

    * most_common(): 모든 요소를 내림차순으로 반환, 인자값을 주면 인자값 보다 높은 항목 반환
```
from collections import Counter
breakfast = ['spam', 'spam', 'eggs', 'spam']
breakfast_counter = Counter(breakfast)
breakfast_counter
# 출력값: Counter({'spam':3, 'eggs':1})

breakfast_counter.most_common()
# 출력값: [('spam', 3), ('eggs', 1)]
breakfast_counter.most_common(1)
# 출력값: [('spam', 3)]
```
 * counter 결합(+) 또는 빼기(-)

```
lunch = ['eggs', 'eggs', 'bacon']
lunch_counter = Counter(lunch)
breakfast_counter + lunch_counter
# 출력값: Counter({'spam':3, 'eggs':3', 'bacon':1})
```
* 공통 인자 얻기(&): Counter의 반환값이 set이기에 가능하다.
* 모든 인자 얻기(|)
```
breakfast_counter & lunch_counter
# 출력값: ({'eggs':1})

breakfast_counter | lunch_counter
# 출력값: ({'spam':3, 'eggs':2, 'bacon':1})
```
* OrderedDict(): 딕셔너리 키값 정렬
```
from collections import OrderedDict
quotes = OrderedDict([
    ('Moe','A wise guy, huh?),
    ('Larry','Ow!'),
    ('Curly','Nyuk nyuk!)
])
for name in quotes:
    print(name) # 키값이 순서대로 출력된다.
```

### deque(데크)
* 큐와 스택의 기능을 모두 가진 큐이며, 시퀀스의 양 끝에 항목을 추가하거나 추출할 때 유용하다.
```
def palindrome(word):
    from collections import deque
    dq = deque(word)
    while len(word) > 1:
        if dq.popleft() != dq.pop(): # 문자열의 왼쪽과 오른쪽 항목이 다른 경우
            return false
    return true
```

### itertools
* 특수 목적의 iterator 함수를 포함하고 있다.

    * chain(): 순회 가능한 인자들을 하나씩 반환
    * cycle(): 인자를 순환하는 무한 iterator
    * accumulate(): 축적된 값을 계산, 두 번째 인자로 함수 전달하여 다른 결과값을 얻을 수 있다.
```
import itertools
for item in itertools.chain([1,2], ['a','b']):
    print(item) # 출력값: 1\n 2\n a\n b\n

for item in itertools.cycle([1,2]):
    print(item) # 출력값: 1\n 2\n 1\n 2\n 무한히 반복

for item in itertools.accumulate([1,2,3,4]):
    print(item) # 출력값: 1\n 3\n 6\n 10\n

def mul(a, b):
    return a * b
for item in itertools.accumulate([1,2,3,4], mul):
    print(item) # 출력값: 1\n 2\n 6\n 24\n
```

### pprint
* print함수 보다 가독성을 높인 상태로 출력한다.
```
from pprint import pprint
quotes = OrderedDict([
    ('Moe', 'A wise guy, huh?'),
    ('Larry', 'Ow!')
])

print(quotes) 
# 출력값: OrderedDict([('Moe', 'A wise guy, huh?'), ('Larry', 'Ow!')])

pprint(quotes)
# 출력값: {'Moe':'A wise guy, huh?', 'Larry':'Ow!}

```