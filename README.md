



## 环境

python3.8

pyyaml、openpyxl、flask、matplotlib、pandas==1.4.2、scikit-learn、statsmodels、pmdarima

TODO: pytorch模型有误，已屏蔽





## 使用

main.py: 调试处理

app.py：接口开启

build.sh：环境检查，后台开启程序脚本

config/config.yaml：配置文件

调用：192.168.xx.xx替换为服务器地址，construction_sites与disposal_sites字段对应excel中数据

```
curl -s -X POST 'http://192.168.xx.xx:5000/algorithm/api/v1/data_prediction' \
-H 'X-Client-Id: client1' \
-H "X-Token: e2311e2f32e4fe81d6f6dfad29b5ab44" \
-H 'Content-Type: application/json' \
-d '{
  "periods": 4,
  "construction_sites": [
    "某某工地项目1",
    "某某工地项目2"
  ],
  "disposal_sites": [
    "某某消纳场1",
    "某某消纳场2"
  ]
}'
```

X-Client-Id有X-Token：服务端app.py中定义

