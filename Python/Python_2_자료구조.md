># 자료구조(Data Structure)
>>## 리스트(List)
>>* 변경 가능(mutable)한 자료구조이다.
>>* 여러 타입을 한 번에 저장할 수 있다.
>>* 리스트 함수들
>>   * append(): 리스트의 끝에 항목 추가(리스트 추가 시 하나의 리스트가 추가된다.)
>>   * extend(): 리스트 병합하기
>>   * insert(a, b): 원하는 위치에 항목 추가하기
>>   * remove(a): 인자값을 통해 특정 값 삭제하기
>>   * pop(): 항목을 가져오는 동시에 그 항목을 리스트에서 삭제하기, 인자가 없을 경우 -1을 사용한다.
>>   * index(): 인자값을 통해 특정 항목의 인덱스 정보 얻기
>>   * in(): 존재여부 확인하기
>>     <pre><code>marxes = ['Groucho', 'Chico', 'Harpo']
>>     'Groucho' in marxes # true 출력</code></pre>
>>   * count(): 특정 값 세기
>>   * sort(): 리스트 자체를 내부적으로 정렬
>>   * sorted(): 정렬된 복사본을 반환, 리스트 함수가 아니다.
>>* 리스트 할당(=)
>>  <pre><code>a = [1,2,3]
>>b = a
>>a[0] = 'first'
>>print(b)
>># 출력값: ['first',2,3], b는 a와 같은 객체에 다른 이름이 부착된 것(공유 O)
>>c = list(a)
>>d = a[:]
>>e = a.copy()
>># c, d, e는 a와 다른 객체에 다른 이름이 부착된 것(공유 X)</code></pre>
>>## 튜플(Tuple)
>>* 불변(immutable)한 자료구조이다.
>>* 튜플 생성 및 변환
>><pre><code># 2가지 튜플 생성 방법 존재
>>empty_tuple = ()
>>one_marx = 'Groucho', # 요소가 한 개일 경우 마지막 콤마가 필요하다.
>>marx_tuple = 'Groucho', 'Chico', 'Harpo'
>>marx_tuple = ('Groucho', 'Chico', 'Harpo')
>>
>># 객체를 튜플로 변환
>>mar_list = ['Groucho','Chico','Harpo']
>>tuple(mar_list)
>># 출력값: ('Groucho','Chico','Harpo')</code></pre>
>>* 튜플 언패킹(tuple unpacking)
>><pre><code>marx_tuple = ('Groucho', 'Chico', 'Harpo')
>>a, b, c = marx_tuple
>># 각각 변수 a, b, c에 순서대로 Groucho, Chico, Harpo값 저장
>></code></pre>
>>## 딕셔너리(Dictionary)
>>* 딕셔너리는 리스트와 다르게 순서를 따지지 않음
>>* 딕셔너리의 키는 불별하기에 리스트, 딕셔너리, 셋이 올 수 없다. 단 튜플은 가능
>>* 딕셔너리 생성 및 데이터 타입 변환
>><pre><code>empty_dict = {}
>>
>>lol = [['a','b'],['c','d'],['e','f']]
>>dict(lol)
>># key: 'a','c','e' value: 'b','d','f'
>>
>>los = ['ab','cd','ef']
>>dict(los)
>># key: 'a','c','e' value: 'b','d','f'</code></pre>
>>* 딕셔너리 병합하기: update()
>><pre><code>first = {'a':1, 'b':2}
>>second = {'b':'platypus'}
>>first.update(second) # 같은 키값이 존재할 경우 병합을 당하는 딕셔너리의 value값이 저장된다.</code></pre>
>>* 키와 del을 통한 특정 항목 삭제 + clear()를 통한 모든 항목 삭제
>><pre><code>del first['a']
>>first.clear()</code></pre>
>>* in을 통한 키값 존재 유무 파악하기
>><pre><code>pythons = {'Chapman':'Graham', 'Clees':'John', 'Jones':'Terry', 'Palin':'Michael'}
>>'Chapman' in pythons # true
>>'Gillian' in ptrhons # false</code></pre>
>>* 딕셔너리 함수들
>>   * get(a, b): 키(a)가 존재하면 그에 대응하는 value값을 키값이 존재하지 않으면 옵션값(b)를 출력한다. 
>>   * keys(): 모든 키 얻기
>>   * values(): 모든 값 얻기
>>   * items(): 모든 쌍의 키-값 얻기
>>* 할당(=) VS 복사: copy()
>><pre><code>signals = {'a':1, 'b':2, 'c':3}
>>save_signals = signals # 할당
>>signals['d'] = 4
>>save_signals # 출력값: {'a':1, 'b':2, 'c':3, 'd':4}
>>
>>save_signals = signals.copy() # 복사
>>signals['d'] = 4
>>save_signals # 출력값: {'a':1, 'b':2, 'c':3}</code></pre>
>>## 셋(Set)
>>* set은 순서가 존재하지 않는다.
>>* set 생성 및 데이터 타입 변환
>><pre><code># 두가지 방법의 set 생성
>>empty_set = set()
>>even_number = {0,2,4,6}
>>
>># 데이터 타입 변환: 중복된 값을 버리고 set으로 변환
>>set('letter') # 출력값: {'l','e','t','r'}
>>set({'apple':'red', 'orange':'orange', 'cherry':'red'})
>># 출력값: {'apple','orange','cherry'} 키에 해당하는 값만 출려된다.</code></pre>
>>* in의 활용
>><pre><code>drinks = {'martini':{'vodka','vermouth'},
>>          'black russian':{'vodka','kahlua'},
>>          'white russian':{'cream','kahlua','vodka'},
>>          'manhattan':{'rye','vermouth','bitters'},
>>          'screwdriver':{'orange juice','vodka'}}
>>
>>for a, b in drinks.items():
>>    if 'vodka' in b:
>>        print(a) # vodka가 포함된 음료수 이름 출력</code></pre>
>>* set의 intersection(교집합)과 union(합집합) 연산자
>><pre><code>for a,b in drinks.items():
>>    if b & {'vermouth','orange juice'}: # 교집합 연산
>>        print(a) # 출력값: martini, manhattan, screwdriver
>>--------------------------------------------------------------------
>>a = {1, 2}
>>b = {2, 3}
>>a & b # 출력값: {2}
>>a.intersection(b) # 출력값: {2}
>>
>>a | b # 출력값: {1,2,3}
>>a.union(b) # 출력값: {1,2,3}
>>
>>a - b # 출력값: {1}
>>a.difference(b) # 출력값: {1}
>>
>>a ^ b # 출력값: {1,3}
>>a.symmetric_difference(b) # 출력값: {1,3}
>>
>># 첫 번째 셋이 두 번째 셋의 subset(부분집합)인가
>>a <= b # 출력값: false
>>a.issubset(b) # 출력값: false
>>a.issuperset(b) # 출력값: false, subset의 반대 개념
>>
>># 첫 번째 셋이 두 번째 셋의 proper subset(진부분집합)인가
>># 두 번째 셋에는 첫 번째 셋의 모든 멤버를 포함한 그 이상의 멤버가 있어야 한다.(동일해서는 안 됨)
>>a < b # 출력값: false
>></code></pre>
>>
>>
>>
>>
>>
