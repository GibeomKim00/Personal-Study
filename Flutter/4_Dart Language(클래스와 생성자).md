# 클래스와 생성자
### 객체 멤버와 클래스 멤버
* 멤버: 클래스에 선언한 변수나 함수
    * 객체 멤버: 생성된 객체를 이용해 접근한다.
    * 클래스 멤버: static 예약어로 선언한 멤버이다.
```
class MyClass {
    String data1 = 'hello';
    static String data2 = 'hello';

    myFun1() {
        print('myFun1 call...');
    }
    static myFun2() {
        print('myFun2 call...');
    }
}

MyClass.data1 = 'world'; # 오류, 객체 멤버로 클래스 이름을 통해 접근 불가
MyClass obj = MyClass();
obj.data1 = 'world';

MyClass.data2 = 'world';
MyClass obj = MyClass();
obj.data2 = 'world'; # 오류, 클래스 멤버는 객체를 통해 접근할 수 없다.
```

### 생성자와 멤버 초기화
* **생성자**(constructor): 클래스에 선언되어 객체가 생성될 때 호출되는 함수이다. 모든 클래스는 생성자를 가지며 개발자가 만들지 않으면 컴파일러가 자동으로 클래스와 같은 이름으로 기본 생성자를 만든다.
* 생성자는 멤버를 초기화하는 용도로 사용된다.
```
class User {
    late String name;
    late int age;
    User(String name, int age) {
        this.name = name;
        this.age = age;
    }
    sayHello() {
        print('name : $name, age : $age');
    }
}
```
```
class User {
    late String name;
    late int age;
    User(this.name, this.age); # 생성자를 단순화한 형태로 나타낼 수 있다.
}
```
* 초기화 목록(initializer list): 생성자 선언부를 콜론(:)으로 구분하여 오른쪽에 작성한다. 리스트에서 특정 항목을 선택하거나 함수 호출로 멤버를 초기화할 때 자주 사용된다.
```
User(String name, int age) : this.name = name, this.age = age {}

class MyClass {
    late int data1;
    late int data2;
    
    MyClass(List<int> args) : this.data1 = args[0], this.data2 = args[1] {}
}
```
```
# 생성자의 초기화 목록이 실행되는 시점은 객체 생성 이전이므로 이곳에서 호출할 수 있는 함수는 static이 추가된 클래스 멤버여야 한다.
class MyClass {
    late int data1;
    late int data2;

    MyClass(int arg1, int arg2) : this.data1 = calFun(arg1), this.data2 = calFun(arg2) {}

    static int calFun(int arg) {
        return arg * 10;
    }
}
```
### **명명된 생성자**(named constructor)
* 한 클래스에 이름이 다른 생성자를 여러 개 선언하는 기법이다. 명명된 생성자는 클래스와 생성자 이름을 점(.)으로 연결해서 작성한다.
```
class MyClass {
    MyClass() {}
    MyClass.first() {}
}

var obj1 = MyClass();
var obj2 = MyClass.first();
```
* this()를 이용해 생성자에서 다른 생성자를 호출할 수 있다. this()는 생성자의 {} 안쪽 본문에 작성할 수 없고 생성자의 콜론(:) 오른쪽 초기화 목록에 작성해야 한다. 그런데 초기화 목록에 this() 호출문을 작성하면 생성자 본문을 작성할 수 없다.
```
class MyClass {
    MyClass(int data1, int data2) {
        print('MyClass() call...');
    }

    MyClass.first(int arg) {
        this(arg, 0); # 오류
    }

    MyClass.first(int arg) : this(arg, 0) {} # 오류

    MyClass.first(int arg) : this(arg, 0) # 성공

    MyClass.second() : this.first(0) # 성공

    # this()를 이용할 때 this() 호출문 외에 다른 구문을 사용할 수 없다.
    MyClass.first(int arg) : this(arg, 0), this.data1=arg1; # 오류
}
```

### 팩토리 생성자(factory constructor)
* factory 예약어로 선언한다. 팩토리 생성자 역시 객체를 생성할 때 호출할 수 있지만, 생성자 호출만으로 **객체가 생성되지 않는다**. 팩토리 생성자에서 적절한 객체를 반환해 줘야 한다.
* 팩토리 생성자는 클래스 외부에서는 생성자처럼 이용되지만 실제로는 객체를 반환하는 **함수**이다.
```
class MyClass {
    factory MyClass() { # 오류, 팩토리 생성자인데 객체를 반환해주지 않는다.
    }

    factory MyClass() {
        return null; # 오류, 팩토리 생성자는 반환 타입을 지정할 수 없으며 클래스 타입으로 고정된다.
                     # 클래스 이름이 MyClass이므로 팩토리 생성자의 반환 타입이 MyClass로 고정된다.
                     # 그리고 MyClass는 널 불허로 선언했으므로 null을 반환할 수 없기에 오류가 발생한다.
    }
}
```
* 팩토리 생성자 자체로는 객체가 생성되지 않고 적절한 객체를 준비해서 반환해 줘야 한다. 따라서 팩토리 생성자가 선언된 클래스에는 객체를 생성하는 다른 생성자를 함께 선언하는 방법을 주로 사용한다.
```
class Image {
    late String url;
    static Map<String, Image> _cache = <String, Image>{};
    Image._instance(this.url);
    factory Image(String url) {
        if (_cache[url] == null) {
            var obj = Image._instance(url); # _instance() 함수는 객체를 생성하는 함수이다.
            _cache[url] = obj;
        }
        return _cache[url]!;
    }
}
```

### 상수 생성자(constant constructor)
* const 예약어로 선언하며 본문을 가질 수 없다. 즉, {}를 추가할 수 없다.
```
class MyClass {
    const MyClass();
}
```
* 상수 생성자가 선언된 클래스의 모든 변수는 final로 선언해야 한다.
```
class MyClass {
    int data1;
    const MyClass(); # 오류
}
```
* 상수 생성자도 객체를 생성할 수 있다.
```
class MyClass {
    final int data1;
    const MyClass(this.data1);
}

main() {
    var obj1 = MyClass(10);
}
```
* const를 추가해 상수 객체를 만들 수 있다. 단, const로 객체를 생성하렴녀 생성자 또한 const로 선언해야 한다.
```
class MyClass {}

main() {
    var obj = const MyClass(); # 오류, const 생성자가 없다.
}
```
```
class MyClass {
    final int data1;
    const MyClass(this.data1);
}

main() {
    var obj = const MyClass(10);
}
```
* 상수 생성자를 선언한 클래스더라도 일반 객체로 선언하면 서로 다른 객체가 생성된다. 그러나 const를 붙여 상수 객체로 선언하면서 생성자에 전달한 값이 똑같으면 객체를 다시 생성하지 않고 이전 값으로 생성한 객체를 사용한다.
```
var obj1 = MyClass(10);
var obj2 = MyClass(10);
print('${obj1 == obj2}') # 출력값: false

var obj1 = const MyClass(10);
var obj2 = const MyClass(10);
print('${obj1 == obj2}') # 출력값: true

var obj1 = const MyClass(10);
var obj2 = const MyClass(20);
print('${obj1 == obj2}') # 출력값: false

var obj1 = const MyClass(10);
var obj2 = MyClass(10);
print('${obj1 == obj2}') # 출력값: false
```