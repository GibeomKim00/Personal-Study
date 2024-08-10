# 함수와 제어문
### 함수 선언과 호출하기
* 다트에서는 함수 오버로딩(함수 이름은 같은데 매개변수가 다른 것)을 제공하지 않는다.
* 함수의 매개변수는 타입을 명시하거나 var로 선언하거나 생략할 수 있다.
```
# var는 대입되는 값에 따라 컴파일 시 타입이 결정된다. 그러나 매개변수에 값이 대입되는 시점은 함수를 호출할 때이므로 컴파일 시 타입을 유추할 수 없다. 따라서 함수 매개변수를 var로 선언 시 dynamic 타입이 된다.
void some(var a) {
    a = 20;
    a = 'world';
    a = true;
    a = null;
}

main() {
    some() # 매개변수에 값을 전달하지 않아서 오류
    some(10)
}
```
```
# 함수의 매개변수 타입을 설정하지 않은 경우 var로 선언한 것과 같다. 즉, dynamic 타입이 된다.
void some1(a) {
    a = 20;
    a = 'world';
    a = true;
    a = null;
}
```
* 함수의 반환 타입도 생략할 수 있으며 이때, dynamic 타입이 된다.
```
some2() {
    return 10;
}

# 반환 값이 없는 경우 null을 반환한다.
some3() {
}
```
* 화살표 함수: 함수의 본문이 한 줄인 경우 중괄호 대신 화살표를 사용하여 나타낼 수 있다.
```
void printUser1() {
    print('hello world');
}

void printUser2() => print('hello world');
```

### 명명된 매개변수
* **명명된 매개변수**는 optional이므로 호출할 때 데이터를 전달하지 않을 수도 있으며, 데이터를 전달할 때는 **'이름: 값'** 형태로 매개변수 이름과 값을 함께 전달한다.
* 중괄호로 묶어서 명명된 매개변수를 사용한다.
* 한 함수에 명명된 매개변수는 한 번만 선언할 수 있으며 순서상 마지막에 선언해야 한다.
* 명명된 매개변수는 기본값을 설정할 수 있다.
```
void some1({String? data2, bool? data3}, int data1) {} # 오류
void some2(int data1, {String? data2, bool? data3}, {int? data4}) # 오류
void some3(int data1, {String? data2, bool? data3}) {}
```
*  명명된 매개변수 호출 규칙
    1. 명명된 매개변수에 데이터를 전달하지 않을 수 있다.
    2. 명명된 매개변수에 데이터를 전달하려면 반드시 이름을 명시해야 한다.
    3. 명명된 매개변수에 데이터를 전달할 때 선언된 순서와 맞추지 않아도 된다.
```
void some(int data1, {String? data2, bool? data3}) {}

some(); # 오류
some(10);
some(10, 'hello', true); # 오류
some(10, data2: 'hello', data3: true)
some(10, data3: true, data2: 'hello')
some(data2: 'hello', 10, data3: true)
```
* 명명된 매개변수에 기본 인자를 설정할 수 있으며, 기본 인자가 없을 경우 널 허용으로 선언해줘야 안전하다.
```
String myFun({String data = 'hello'}) {
    return data;
}

main() {
    print(${myFun()}); # 출력값: hello
    print(${myFun(data: "world")}); # 출력값: world
}
```
* required - 명명된 매개변수에서 required 예약어는 반드시 값을 전달받도록 한다.
```
someFun({required int arg1}) {
    print('someFun().. arg1 : $arg1');
}

main() {
    someFun() # 오류
    someFun(arg1: 10);
}
```

### 옵셔널 위치 매개변수(optional positional parameter)
* 명명된 매개변수 처럼 인자를 전달받지 않을 수 있다. 그러나 값을 전달할 때 이름 대신 매개변수가 선언된 순서(위치)에 맞게 호출해야 한다.
* 매개변수들을 대괄호 []로 묶는다.
* 함수의 마지막 매개변수에만 사용할 수 있다.
* 매개변수에 기본 인자를 설정할 수 있다.
* 옵셔널 위치 매개변수 호출 방법
    1. 매개변수 이름은 생략한다.
    2. 매개변수가 선언된 순서에 따라 값이 할당된다.
```
void some(int arg1, [String arg2 = 'hello', bool arg3 = false]) {}

some(); # 오류
some(10)
some(10, arg2: 'world', args: true) # 오류
some(10, 'world', true)
some(10, true, 'world') # 오류
some(10, 'world')
some(10, true) # 오류
``` 

### 함수 타입 인수
* **함수 타입(function type)**: 함수도 객체이며, 함수를 대입할 수 있는 객체를 함수 타입이라고 하며 
**Function**으로 선언한다.
```
int plus(int no) {
    return no + 10;
}
int multiply(int no) {
    return no * 10;
}

# 매개변수로 함수를 전달받고 함수를 반환한다.
Function testFun(Fucntion argFun) {
    print('argFun: ${argFun(20)}');
    return multiply;
}

main(List<String> args) {
    var result = testFun(plus);
    print('result: ${result(20)}');
}
```
* 함수 타입 변수에 대입할 함수를 특정한 형태로 한정할 수도 있다.
```
# 매개변수와 반환값이 int 타입인 함수만 받을 수 있다.
some(int f(int a)) {
    f(30)
}

main(List<String> args) {
    some((int a) {
        return a + 20;
    });
}
```
* 익명 함수(anonymous functions): 이름이 생략된 함수로 **람다 함수(lambda function)**라고 부른다.
```
Function fun2 = (arg) {
    return 10;
}
```

### getter와 setter 함수
* 데이터를 가져올 함수에 **get** 예약어를, 데이터를 변경할 함수에 **set** 예약어를 추가하면 함수를 변수처럼 이용할 수 있다.
* get 예약어를 추가한 함수는 데이터를 가져오는 역할이기에 매개변수 부분을 선언할 수 없다.
```
String _name = 'Hello';

# 두 함수의 이름이 동일하다.
String get name {
    return _name.toUpperCase();
}

set name(value) {
    _name = value;
}

# 함수를 변수처럼 사용 가능하다.
main(List<String> args) {
    name = 'world'; # setter 함수 호출
    print('name: $name') # getter 함수 호출
}
```

### 기타 연산자
1. ~/ : 나누기 연산자
    * / 는 실수를 반환, ~/ 는 정수를 반환한다.

2. is, as : 타입 확인과 변환
    * is 연산자는 타입을 확인해 true 또는 false 반환
    * as 연산자는 타입을 바꿔준다.
```
class User {
    void some() {
        print('User...some()...');
    }
}

main() {
    Object obj = User(); # Object은 다트에서 최상위 클래스로 하위에서 상위로 자동으로 형 변환이 일어난다.
    obj.some(); # 오류

    # is 연산 결과 true이기에 obj 객체가 User 타입으로 형 변환 된다.
    if(obj is User) {
        obj.some(); # 정상
    }

    Object obj1 = User();
    (obj1 as User).some(); # 상위에서 하위로 형 변환은 명시적 형 변환을 해줘야 하며 이때, as 연산자를 사용
}
```

3. .., ?.. : 반복해서 접근하기
    * 같은 객체를 반복해서 접근할 때 편리한 캐스케이드 연산자이다. nullable 객체일 때 ..?를 사용
```
class User{
    String? name;
    int? age;

    some() {
        print('name: $name, age: $age');
    }
}

var user = User();
user.name = 'kkang';
user.age = 10;
user.some();

# 객체 이름 생략 가능
User()
    ..name = 'kkang'
    ..age = 30
    ..some();
```

### 제어문
1. for 반복문에서 in 연산자
```
main() {
    var list = [10, 20, 30];

    for(var i = 0; i < list.length; i++) {
        print(list[i]);
    }

    for(var x in list) {
        print(x);
    }
}
```
2. switch ~ case 선택문
    * 맨 마지막 case 문을 제외하고 break, continue, return, throw중 하나를 작성해여 실행 흐름을 결정해 줘야 한다.
    * break: switch 문 나가기
    * continue: 특정 위치로 이동하기
    * return: switch 문이 작성된 함수 종료하기(반환하기)
    * throw: switch 문이 작성된 함수 종료하기(던지기)
```
some(arg) {
    swtich(arg) {
        case 'A': # 오류
            print('A'):
        case 'B':
            print('B');
    }
}
```
```
some(arg) {
    swtich(arg) {
        case 'A':
            print('A'):
            break;
        case 'B':
            print('B');
    }
}
```
3. 예외 던지기와 예외 처리
    * 예외를 던지는 throw 문: throw 오른쪽에는 객체를 작성한다.
    ```
    some() {
        throw Exception('my exception');
    }

    # some 함수를 호출한 곳에 해당 문자열을 던진다.
    some() {
        throw 'my exception';
    }
    ```
    * try~on~finally 예외 처리: try 문에 작성한 코드에서 예외가 발생하면 on 문이 실행되며, finally 문에는 예외와 상관없이 무조건 실행할 코드를 작성한다. catch 문으로 예외 객체를 받을 수 있다.
    ```
    some() {
        throw FormatException('my exception')
    }
    main(List<String> args) {
        try {
            print('step1');
            some();
            print('step2');
        } on Exception {
            print('step3');
        } on FormatException catch(e) {
            print('step4 $e');
        } finally {
            print('step5');
        }
        print('step6');
    }

    # 출력값: step1 step4 FormatException: my exception step5 step6
    ```
    ```
    # 예외 종류 구분 없이 간단하게 작성할 수도 있다.
    try {
        some();
    } catch(e) {
        print('catch...$e');
    }
    ```