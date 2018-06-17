import json


dict1 = { 'name' : 'song', 'age' : 10 }
j1 = {"name":"홍길동", "birth":"0525", "age": 30}
json_string = json.dumps(j1)

#json 파일 저장
with open('../Data/Sample.json',mode='w') as f:
    json.dump(json_string,f,indent=2)

#json 파일 읽어오기
with open('../Data/Sample.json', mode='r') as f:
    json_string = json.load(f)

print(dict1['name'])
print(dict1['age'])

dicToJson = json.dumps(dict1)
print(dicToJson)
print(type(dicToJson))
print(json_string)

print("from json to dictionary")

dict2 = json.loads(dicToJson)

j1['age'] += 1

print(j1['age'])

print(dict2)
print(type(dict2))