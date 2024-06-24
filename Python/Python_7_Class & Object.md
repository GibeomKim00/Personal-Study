# Class & Object
### Class
* 객체를 만들기 위한 틀
```
class Person(): # 빈 클래스 정의
    pass

someone = Person() # 객체 생성
```
* initialize method(객체 초기화 메서드)

    * 객체 생성 시 호출되는 메서드이다.
    * 첫 매개변수는 self여야 한다.
```
class Person():
    def __init__(self, name):
        self.name = name

hunter = Person('Elmer Fudd')
# 새롭게 생성된 객체를 self에 전달, 인자를 name에 전달한다.
# 객체에 name값 저장
# 새로운 객체를 반환 후 hunter에 이 객체를 연결한다.
```
* 속성(attribute)

* 메서드(method)

### 상속(inheritance)
* 기존 클래스의 코드를 재사용할 수 있다.
* parent class = super class = base class
* child class = sub class = derived class
```
class Car():
    def exclaim(self):
        print("I'm a Car!")

class Yugo(Car): 
    pass

give_me_a_car = Car()
give_me_a_yugo = Yugo() # Yugo class의 instance이지만 Car class의 기능도 가능하다.
```
* override: 상속 받은 method의 기능을 변경
```
class Car():
    def exclaim(self):
        print("I'm a Car!")

class Yugo(Car): 
    def exclaim(self):
        print("I'm a Yugo! Much like a Car, but more Yogu-ish.")

```
* super: 자식 클래스에서 부모 클래스의 메서드를 호출한다.
```
class Person():
    def __init__(self, name):
        self.name = name

class EmailPerson(Person):
    def __init__(self, name, email): # __init__ 함수 override
        super().__init__(name)  # override하는 순간 부모 클래스의 __init__ 함수는 호출되지 않기에 우리가 명시적으로 호출해야 한다.
        self.email = email
```
### self
* 객체를 생성하고 객체의 메서드를 호출할 때 객체 본인을 첫 번째 매개변수로 전달한다.

### get/set과 property/decorator
* getter와 setter 메서드를 통해 객체의 속성값에 마음대로 접근하지 못하게 한다.
* getter: 데이터를 읽어주는 메서드
* setter: 데이터를 변경해주는 메서드
* property: 파이썬 내장 함수로, 첫 번째 인자로 getter함수를 두 번째 인자로 setter함수를 넘겨준다.
```
class Duck():
    def __init__(self, input_name):
        self.hidden_name = input_name
    def get_name(self):
        print('inside the getter')
        return self.hidden_name
    def set_name(self, input_name):
        print('inside the setter')
        self.hidden_name = input_name
    name = property(get_name, set_name) # name을 속성 처럼 접근할 수 있다.

fowl = Duck('Howard')
fowl.name
# 출력값
inside the getter
Howard

fowl.name = 'Daffy'
# 출력값
inside the setter
```
* getter method 앞에 @property decorator를 setter method 앞에 @name.setter decorator를 쓴다.
```
class Duck():
    def __init__(self, input_name):
        self.hidden_name = input_name
    @property
    def name(self):
        print('inside the getter')
        return self.hidden_name
    @name.setter
    def name(self, input_name):
        print('inside the setter')
        self.hidden_name = input_name

# name을 속성(attribute)처럼 접근할 수 있다.
fowl = Duck('Howard')
fowl.name
# 출력값
inside the getter
Howard

fowl.name = 'Donald'
# 출력값
inside the setter
```
```
# property를 통해 계산된 값을 참조할 수 있으며 여기서도 속성처럼 메서드를 참조할 수 있다.

class Circle():
    def __init__(self, radius):
        self.radius = radius
    @property
    def diameter(self):
        return 2 * self.radius

c = Circle(5)
c.radius # 출력값: 5
c.diameter # 출력값: 10
c.diameter = 10 # 오류 발생, 일기 전용으로만 만들었기 때문에

    @ diameter.setter
    def diameter(self, radius):
        self.radius = radius
c.diameter = 10 # 정상 작동
```

### private
* python에서는 속성 앞에 두 언더스코어(__)를 붙이면 외부로부터 속성에 접근할 수 없게 된다.
```
class Duck():
    def __init__(self, input_name):
        self.__name = input_name
    @property
    def name(self):
        print('inside the getter')
        return self.__name
    @name.setter
    def name(self, input_name):
        pritn('inside the setter')
        self.__name = input_name
```

### method type
* instance method: 첫 번째 매개변수가 self인 메서드로 파이썬은 인스턴스 메서드를 호출할 때 객체를 전달한다.
* class method: 클래스에 대한 어떤 변화는 모든 객체에 영향을 미친다.

    * @classmethod 데커레이터가 있으면 class method이다.
    * 첫 번째 매개변수는 클래스 자신이다.
```
class A():
    count = 0
    def __init__(self):
        A.count += 1 # 클래스 속성
    def exclaim(self):
        print("I'm an A!")
    @classmethod
    def kids(cls):
        print("A has", cls.count, "little objects.")
```
* static method(정적 메서드)

    * @staticmethod 데커레이터가 있으면 static method이다.
    * 첫 번째 매개변수로 self 또는 cls가 없으며, 객체 없이 호출이 가능하다.
```
class CoyoteWeapon():
    @staticmethod
    def commercial():
        print('This CoyoteWeapon has been brought to you by Acme')

Coyoteweapon.commercial()
# 출력값: This CoyoteWeapon has been brought to you by Acme
```

### 특수 메서드
* 비교 연산 특수 메서드(양쪽 끝에 두 개의 언더스코어 붙이기)

    * eq(self, other): self == other
    * ne(self, other): self != other
    * lt(self, other): self < other
    * gt(self, other): self > other
    * le(self, other): self <= other
    * gt(self, other): self >= other
*  산술 연산 특수 메서드
    * add(self, other): self + other
    * sub(self, other): self - other
    * mul(self, other): self * other
    * floordiv(self, other): self // other
    * truediv(self, other): self / other
    * mod(self, other): self % other
    * pow(self, other): self ** other
* 기타 특수 메서드
    * str(self): str(self)
    * repr(self): repr(self)
    * len(self): len(self)
```
class word():
    def __init__(self, text):
        self.text = text
    def __eq__(self, word2):
        return self.text.lower() == word2.text.lower()
    def __str__(self):
        return self.text
    def __repr__(self):
        return "Word('" + self.text + "')"

first = Word('ha')
first # __repr__ 호출
print(first) # __str__ 호출
```

### namedtuple
* tuple의 sub class로 모듈을 불러와 사용 가능하다.
* namedtuple을 딕셔너리의 key 처럼 사용할 수 있다.
* namedtuple은 불변하다
```
class Bill():
    def __init__(self, description):
        self.description = description

class Tail():
    def __init__(self, length):
        self.length = length

class Duck():
    def __init__(self, bill, tail):
        self.bill = bill
        self.tail = tail

from collections import namedtuple
Duck = namedtuple('Duck', 'bill tail') # Duck이 namedtuple로 스페이스로 구분된 필드 이름(bill과 tail)의 문자열
duck = Duck('wide orange', 'long')
duck # 출력값: Duck(bill='wide orange', tail='long)
duck.bill # 출력값: wide orange
duck,tail # 출력값: long


# 딕셔너리를 통해 namedtuple을 만들 수 있다.
parts = {'bill':'wide orange', 'tail':'long'}
duck2 = Duck(**parts) # **는 키워드 인자로 딕셔너리의 키와 값을 추출하여 Duck() 인자로 제공한다.
duck2 # 출력값: Duck(bill='wide orange', tail='long)
```