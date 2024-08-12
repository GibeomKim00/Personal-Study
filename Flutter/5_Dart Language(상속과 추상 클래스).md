# 상속과 추상 클래스
### 상속과 오버라이딩
* 상속(inheritance): extends 예약어를 통해 클래스 상속을 받는다.
* 오버라이딩(overriding): 상속 받은 멤버를 재정의하여 사용하는 것으로 오버라이딩을 하면 부모 클래스에 선언된 멤버가 자식 클래스에 상속되지 않는다.
```
class SuperClass {
    int myData = 10;
    void myFun() {
        print('Super...myFun()...');
    }
}

class SubClass extends SuperClass {
    int myData = 20;
    void myFun() {
        print('Sub...myFun()...');
    }
}

main(List<String> args) {
    var obj = SubClass();
    obj.myFun(); # 출력값: Sub...myFun()...
    print('${obj.myData}') # 출력값: 20
}
```
* super 예약어를 통해 부모 클래스에 선언된 똑같은 이름의 멤버에 접근할 수 있다.
```
class SubClass extends SuperClass {
    int myData = 20;
    void myFun() {
        super.myFun()
        print('${myData}, ${super.myData}');
    }
}

main(List<String> args) {
    var obj = SubClass();
    obj.myFun(); # 출력값: Super...myFun()... 20, 10
}
```
* **부모 생성자 호출**: 자식 클래스의 객체를 생성할 때 자식 클래스의 생성자가 호출되는데, 이때 부모 클래스의 생성자도 반드시 호출되게 해야한다. super() 문을 이용하여 부모 클래스의 생성자를 호출할 수 있다.
    * 부모 클래스의 생성자가 기본 생성자인 경우 super()문을 사용하지 않아도 컴파일러가 자동으로 부모 클래스의 생성자를 호출하기에 오류가 발생하지 않는다.
    * 단, 부모 클래스의 생성자에 매개변수가 있거나 명명된 생성자인 경우 super()문을 생략하면 안 된다.
```
class SuperClass {
    SuperClass() {}
}

class SubClass extends SuperClass {
    SubClass() : super() {}
}
```
```
class SuperClass {
    SuperClass(int arg) {}
    SuperClass.first() {}
}

class SubClass extends SuperClass {
    SubClass() : super() {} # 오류, 부모 클래스의 어떤 생성자를 호출해야 할지 컴파일러는 모른다.
    SubClass() : super(10) {}
    SubClass.name() : super.first() {}
}
```
* 부모 클래스 멤버 초기화하는 법
```
class SuperClass {
    String name;
    int age;
    SuperClass(this.name, this.age) {}
}

class SubClass extends SuperClass {
    SubClass(String name, int age) : super(name, age) {}

    # 또는 더 간단하게
    SubClass(super.name, super.age);
}
```

### 추상 클래스와 인터페이스
* **추상 클래스**: 추상 함수만 제공하여 상속받는 클래스에서 반드시 재정의해서 사용하도록 강제하는 방법이다. 이때, 추상 함수란 실행문이 작성된 본문 {}이 없는 함수를 의미한다. 추상 함수는 추상 클래스에서만 작성할 수 있다.
```
class User {
    void some(); # 오류, 일반 클래스에 추상 함수 선언 불가
}

abstract class User {
    void some();
}
```
* 추상 클래스를 상속받은 자식 클래스는 추상 함수를 반드시 재정의해 줘야 한다.
```
abstract class User {
    void some();
}

class Customer extends User {
    @override
    void some() {}
}
```

* **인터페이스(interface)**: 부모의 특징을 도구로 사용해 새로운 특징을 만들어 사용하는 객체지향 프로그래밍 방법이다.
    * 자바에서는 interface 예약어로 인터페이스를 선언하고, inplements 예약어로 인터페이스를 구현하는 클래스를 선언하는데 다트 언어에서는 interface 예약어가 없다.
    * 다트 언어에서는 모든 클래스가 암시적 인터페이스(implicit interface)이다.
    * implements 예약어를 통해 인터페이스를 구현해야 하는 클래스는 인터페이스에 선언된 멤버를 구현 클래스에서 모두 재정의해 줘야 한다.
```
class User {
    int no;
    String name;

    User(this.no, this.name);
    String greet(String who) => 'Hello, $who. I am $name. no is $no';
}

class MySubClass extends User {
    MySubClass(super.name, super.no);
}

class MyClass implements User {
    int no = 10;
    String name = 'kim';
    @override
    String greet(String who) {
        return 'hello';
    }
}

# 하나의 클래스에 여러 인터페이스를 지정해서 선언할 수 있다.
class MyClass implements User, MyInterface {
}

# 구현 클래스의 객체는 인터페이스 타입으로 선언할 수 있다.
User user = MyClass();
print('${user.greet('lee')}');
```

### 믹스인
* mixin 예약어로 선언하며, 변수와 함수를 선언할 수 있지만 클래스가 아니기에 생성자는 선언할 수 없다. 믹스인은 생성자를 가질 수 없기에 객체를 생성할 수 없다.
* 다트 언어에서는 다중 상속을 지원하지 않기에 믹스인을 통해 **여러 클래스에 선언된 멤버를 상속한 것처럼 이용**할 수 있다. 이때, with 예약어를 사용한다.
```
mixin MyMixin {
    int mixinData = 10;
    void mixInFun() {
        print('MyMixin... mixInFun()...');
    }
}

class MySuper {
    int superData = 20;
    void superFun() {
        print('MySuper...superFun()...');
    }
}

class MyClass extends MySuper with MyMixin {
    void sayHello() {
        print('$superData, $mixinData'); # 출력값: 20, 10
        mixInFun(); # 출력값: MyMixin...mixInFun()...
        superFun(); # 출력값: MySuper...superFun()...
    }
}
```
* with 예약어로 믹스인을 이용한 클래스의 객체는 믹스인 타입으로 사용할 수 있다.
```
class MyClass with MyMixin {}

main() {
    var obj = MyClass();

    if (obj is MyMixin) {
        print('obj is MyMixin');
    } else {
        print('obj is not MyMixin);
    }

    MyMixin obj = MyClass();
}
```
* on 예약어를 통해 특정 타입의 클래스에서만 사용하도록 제한할 수도 있다.
```
mixin MyMixin on MySuper {
}

class MySuper {
}
class MyClass extends MySuper with MyMixin { # 성공
}
class MySomeClass with MyMixin { # 실패
}
```
* 클래스도 with 예약어를 사용할 수 있지만, with 예약어를 사용할 클래스에는 생성자를 선언할 수 없다.(믹스인은 객체를 생성할 수 없기에)
```
class User() {
}

class MyClass with User {
}

# 아래는 오류 발생
class User() {
    User() {}
}

class MyClass with User { # 오류
}
```