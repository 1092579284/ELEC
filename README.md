### 步骤

1、运行`data_preparation.py`，自动下载苹果和微软的股票dataset

2、运行`train_model.py`，训练模型，并且保存

3、运行`server.py`服务器，下面的指令在终端运行，路径要改一下

```
cd C:\Users\Administrator\Desktop\ELEC && Start-Process powershell -ArgumentList "-NoExit", "-Command", "python server.py"
```



4、运行`client.py`客户端，输入apple（aapl）或者microsoft（msft），得到预测结果，下面是示例

```
Hello, I'm the Oracle. How can I help you today?

You: microsoft
Oracle: Based on my analysis, MSFT's stock price tomorrow will be approximately $439.43

You: apple
Oracle: Based on my analysis, AAPL's stock price tomorrow will be approximately $229.38

You: nvidia
Oracle: I can only predict Apple (AAPL) or Microsoft (MSFT) stock prices.
```





