# Dart Language
## 기본 기능 알아보기
* main() 함수

    * 다트 엔진은 main 함수를 진입점(entry point)로 삼는다. 따라서 다트 엔진은 main() 함수를 호출하면서 프로그램이 실행된다.

* import - 라이브러리 불러오기
```
# flutter_lab 프로젝트에 lib 폴더와 some_folder가 있다고 가정해보자.
# lib 폴더에는 section1/main.dart과 outer_folder/outer_main.dart 가 있고,
# some_folder 폴더에는 some_main.dart 가 있다.

# lib 내부에 있는 다른 파일은 상대 경로로 불러올 수 있지만, 외부에 있는 파일은 상대경로로 불러올 수 없다.
import '../outer_folder/outer_main.dart'; # 성공
import '../../some_folder/some_main.dart'; # 오류

# package 접두사로 불러오기
import 'package:flutter_lab/some_folder/some_main.dart';

# dart 접두사로 불러오기 - 다트 언어에서 기본으로 제공하는 라이브러리를 불러올 때 사용한다.
import 'dart:core';
import 'dart:async';
```

* 접근 제한자(access modifier)

    * 다른 언어는 public, private 등을 통해 접근을 제한할 수 있지만 dart 언어는 접근 제한자가 없으며 기본으로 public 상태가 된다.
    * 그러나 선언된 구성 요소를 다른 다트 파일에서 사용하지 못하게 접근 범위를 한정해줄 수 있다.
```
int no1 = 10;
int _no2 = 20; # 현재 파일에서만 사용 가능

void sayHello1() {}
void _sayHello2() {}

class User1 {}
class _User2 {}
```

* as - 별칭
```
import 'test1.dart' as Test1;

main() {
    no1 = 20; # 오류 발생, test1.dart 파일에 선언된 요소를 사용하기 위해서는 반드시 별칭이 필요하다.
    Test1.no1 = 30;
}
```

* show - 특정 요소 불러오기
```
import 'test1.dart' show no1, User1;

main() {
    no1 = 30;
    User1 user1 = User1();

    sayHello1(); # 오류 발생
}
```

* hide - 특정 요소 제외하기
```
import 'test1.dart' hide sayHellow1, User1;

main() {
    no1 = 30;
    sayHellow1(); # 오류 발생
    User1 user1 = User1(); # 오류 발생
}
```

## 라이브러리 만들기
* 여러 개의 파일을 사용할 경우 일일이 import할 수 없다.
```
# a.dart 파일
int aData = 10;

# b.dart 파일
int bData = 20;

# 다른 다트 파일
import 'a.dart';
import 'b.dart';

main() {
    print('$aData, $dData');
}
```

* part, part of 예약어를 이용해 라이브러리로 묶으면 한 번의 import로 모든 다트 파일을 사용할 수 있다.
```
# a.dart 파일
part of my_lib;
int aData = 10;

# b.dart 파일
part of my_lib;
int bData = 20;

# myLib.dart 파일
library my_lib;
part 'a.dart';
part 'b.dart';

# 다른 다트 파일
import 'myLib.dart';

main() {
    print('$aData, $dData');
}
```

## 데이터 타입
* 다트 언어에서 <span style="color:#ffd33d">**모든 변수는 객체**</span>이다.

* 다트의 타입 클래스
    1. dart:core library
        * bool(true, false)
        * double(실수)
        * int(정수)
    2. X
        * num(숫자, double과 int의 상위 클래스)
        * String(문자열)
    3. dart:typed_data
        * ByteData(바이트)

* 문자열 템플릿: 문자열에 동적인 결과를 포함하는 것
```
main() {
    int no = 10;
    String name = 'kkang';

    String myFun() {
        return 'kim';
    }

    print('no: $no, name : $name, 10 + 20 : ${10 + 20}, myFun() : ${myFun()}')
}
```

* 형 변환: 타입의 변수가 모두 객체이기에 자동으로 형 변환이 발생하지 않는다. 형 변환을 위해서 함수를 사용해야 한다.
```
int n1 = 10;
double d1 = 10.0;
String s1 = '10';

double d2 = n1.toDouble();
int n2 = d1.toInt();
String s2 = n1.toString();
int n3 = int.parse(s1);
```

* 상수 변수: 초기값 대입 후 값 변경 불가
    1. const - 컴파일 타임 상수 변수
        * 컴파일 단계에서 상수가 되므로 처음 초기값을 주지 않으면 오류가 발생한다.
        * 클래스에 선언할 때 **static** 변수로만 선언할 수 있다.
        ```
        const String data1; # 오류

        class User {
            static const String data2 = 'hello';

            void some() {
                const String data3 = 'hello';

                data1 = 'world' # 오류
            }
        }
        ```

    2. final - 런 타임 상수 변수
        * const와 같은 상수이기에 중간에 값 변경은 불가하지만, 처음 초기값을 주지 않아도 된다.
        ```
        final int no1;

        class MyClass {
            final int no2;

            void() {
                final no3;
                no3 = 10;
                no3 = 20; # 오류
            }
        }
        ```
    **주의**: const 예약어로 선언된 변수에 문자열 템플릿을 통해 값을 대입하고 싶으면 그 값도 컴파일 타임 상수(const)로 선언돼있어야 한다.
    ```
    String s1 = 'hello';
    const String s2 = 'world';
    final String s3 = 'helloworld';

    String s4 = '$s1, $s2';
    const string s5 = '$s1, $s2, $s3'; # s1, s3로 인한 오류 발생
    final string s6 = '$s1, $s2, $s3';
    ```

* var, dynamic 타입: 타입을 유추하거나 모든 타입의 데이터를 대입할 수 있는 타입
    1. var - 타입 유추
        * 대입하는 값에 따라 결정된다. 즉, 컴파일러가 대입하는 값을 해석해 타입을 추측
        ```
        var no1 = 10; # 변수 타입은 int
        no1 = 20;
        no1 = 'hello'; # 오류 발생
        ```
        * 선언과 동시에 값을 대입하지 않으면 dynamic 타입이 된다.
        ```
        var no2;
        no2 = 20;
        no2 = 20.0
        no2 = 'hello'
        ```
    2. dynamic - 모든 타입 지원(Nullable을 포함)


* 컬렉션 타입: 한 변수에 여러 데이터를 저장할 수 있는 방법으로 배열(Array), 리스트(List), 집합(Set), 맵(Map)등이 있다.
    1. List
        * 데이터를 저장 후 인덱스값으로 데이터를 이용할 수 있는 컬렉션 타입의 클래스이다.
        ```
        List list1 = [10, 'hello']; # List 타입으로 선언했지만 리스트에 대입할 데이터 타입을 지정하지 
                                     않았기에 dynamic 타입 리스트가 된다.
        print('List: [${list1[0]}, ${list1[1]}]');
        ```
        * generic을 통해 리스트에 저장 가능한 데이터 타입을 제한할 수 있다.
        ```
        List<int> list2 = [10, 20, 30];
        list2[0] = 'hello'; # 오류
        ```
        * 데이터 추가 및 제거
        ```
        List<int> list2 = [10];
        list2.add(20); # [10, 20]
        list2.removeAt(0); # [20]
        ```
        * 리스트 크기 제한: 리스트 클래스에 선언된 생성자 filled(), generate() 이용
        ```
        # 첫 번째 인자는 리스트의 크기, 두 번째 인자는 초기화 값에 해당한다.
        var list3 = List<int>.filled(3, 0);
        print(list3); # [0, 0, 0]
        list3.add(10); # 런 타임 오류

        # 리스트의 제한된 크기보다 더 많은 데이터를 집어넣고 싶으면 filled() 생성자의 growable 매개변수 true로 지정
        var list3 = List<int>.filled(3, 0, growable: true);
        list3.add(10); # 정상 작동

        # generate() 생성자가 초기값을 지정해주는 두 번째 매개변수에 함수 지정이 가능
        var list4 = List<int>.generate(3, (index) => index * 10, growable: true);
        print(list4) # 출력값: [0, 10, 20]
        ```

    2. Set
        * 컬렉션 타입의 클래스이며, 인덱스를 통해 데이터에 접근 가능하다.
        * 중복 데이터를 허용하지 않는다.
        ```
        Set<int> set1 = {10, 20, 10};
        print(set1); # 출력값: {10, 20}
        set1.add(30);
        print(set1); # 출력값: {10, 20, 30}

        Set<int> set2 = Set(); # 빈 set
        ```
    3. Map
        * 데이터를 키와 값의 형태로 저장한다.
        * 키를 통해 데이터에 접근한다.
        ```
        Map<String, String> map1 = {'one':'hello', 'two':'world'};

        print(map1['one']);
        map1['one'] = 'world';
        ```

## 널 포인트 예외
* 널 안전성: 널 포인트 예외(null point exception)를 프로그램을 실행하기 전 코드를 작성한 시점에 점검하는 것
    * 널 포인트 예외: 객체가 특정 값이 아닌 널을 가리켜 생기는 오류로 컴파일러가 색출하지 못한다.

* 널 허용(Nullable)과 널 불허(NonNull): 컴파일러에 널을 대입할 수 있는지 없는지를 명확하게 알려 줘야 한다.
    * 다트 언어에서 변수는 기본으로 널 불허로 선언된다.
    * 널 허용으로 선언하고 싶으면 타입 뒤 물음표(?)를 추가해준다.
    ```
    int a1 = 10; # 널 불허 변수
    int? a2 = 10; # 널 허용 변수

    a1 = null; # 오류
    a2 = null; 
    ```


* 널 불허 변수 초기화: 모든 변수는 객체이며, 변수 선언과 동시에 초기값을 주지 않으면 자동으로 널로 초기화 된다. 이때, 널 불허 변수에 초기값을 주지 않으면 오류가 발생한다.
```
int a1; # 오류
int? a2;
```
* 단, 톱 레벨에 선언된 변수와 클래스의 멤버 변수에만 초기값을 설정해준다. 지역 변수에는 널 불허 변수에 초기값을 설정하지 않아도 되지만 사용하기 전에는 값을 무조건 대입해야 한다. 
```
int a1; # 오류

class User {
    int a1; # 오류
}

testFun() {
    int a1; # 성공
    a1 = null; # 오류

    print(a1 + 10); # 오류
}
```

* var 타입의 널 안전성
    * var로 선언된 변수는 대입하는 값에 따라 타입이 결정된다. 따라서 널 허용 여부도 대입되는 값에 따라 컴파일러가 자동으로 결정한다.
    * var 뒤에 물음표(?)를 붙일 수 없다.
```
var a1 = 10; # int 타입
var a2 = null; # dynamic 타입
var a3; # dynamic 타입
var? a4 = null; # 오류

a1 = null; # 오류
a2 = 10;
a2 = null;
a3 = 'hello';
a3 = null;
```

* 널 안전성과 형 변환
    * Nullable과 Nonnull은 타입(클래스)이다.
    * Nullable은 Nonnull의 상위 클래스이다. 따라서 널 불허 변수를 널 허용 변수에 대입 시 자동 형 변환이 일어난다.
    ```
    int a1 = 10;
    int? a2 = 10;
    
    a2 = a1; # 성공
    a1 = a2; # 오류
    ```
    * 명시적 형 변환 연산자(as)를 통해 널 허용 변수를 널 불허 변수에 대입할 수 있다.
    ```
    int a1 = 10;
    int? a2 = 20;
    
    a1 = a2 as int;
    print("a1: $a1, a2: $a2"); # 출력값: a1: 20, a2: 20
    ```

* late 연산자 - 초기화 미루기
    * 널 불허 변수를 널인 상태로 이용하다가 앱이 실행될 때 값을 결정할 수 있다.
    ```
    int a1; # 오류
    late int a2; # 성공

    print('${a2 + 10}'); # 오류
    a2 = 20;
    print('${a2 + 10}');
    ```

## 널 안전성 연산자
1. ! 연산자 - 널인지 점검할 때
    * 변숫값이 널인 경우 런 타임 오류 발생
    ```
    int? a1 = 20;

    a1!;
    a1 = null;
    a1! # 런 타임 오류
    ```

    * 함수 호출 구문에도 사용 가능하다. 변수든 구문이든 결과가 널인 경우 런 타임 오류 발생
    ```
    int? some(arg) {
        if (arg == 10) {
            return 0;
        } else {
            return null;
        }
    }

    main() {
        int a = some(10)!;
        int b = some(20)!; # 런 타임 오류
    }
    ```

2. ?. 또는 ?[] 연산자 - 멤버에 접근할 때
    * 널 허용 객체 또는 리스트의 멤버에 접근할 시 위 연산자 사용
    * ?. : 객체가 널이 아닐 때만 멤버에 접근하며, 널이면 멤버에 접근할 수 없고 null을 반환한다.
    ```
    String? str = 'hello';

    str.isEmpty; # 오류, 널 허용 변숫값이 널일 수 있기에 ?. 연산자를 붙이라는 오류 발생
    ```
    ```
    int? no1 = 10;
    bool? result1 = no1?.isEven;
    print(result1); # 출력값: true

    no1 = null;
    bool? result2 = no1?.isEven; # 객체가 널이기에 멤버에 접근 불가
    print(result2) # 출력값: null
    ```

    * ?[] : 널 허용 리스트의 데이터를 인덱스로 접근할 때 사용
    * List 객체가 널이 아닐 때는 데이터에 접근 가능하며, 널인 경우 null 반환
    ```
    List<int>? list = [10, 20, 30];
    print('list[0] : ${list?[0]}'); # 출력값: list[0] : 10
    list = null;
    print('list[0] : ${list?[0]}'); # 출력값: list[0] : null
    ```

3. ??= - 값을 대입할 때
    * 널 허용 변수에 널이 아닌 값만 대입하고 싶을 때 사용
    * 오른쪽 값이 널이 아닌 경우 대입, 널인 경우 대입하지 않는다.
    ```
    int? data3;
    data3 ??= 10;
    print(data3) # 출력값: 10
    data3 ??= null;
    print(data3) # 출력값: 10
    ```

4. ?? - 값을 대체할 때
    * 널 허용 변수가 널일 때 대체할 값을 지정
    ```
    Strin? data4 = 'hello';
    String? result = data4 ?? 'world';
    print(result); # 출력값: hello

    data4 = null;
    result = data4 ?? 'world';
    print(result); # 출력값: world
    ```