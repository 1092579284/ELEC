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





1. 网页都改成英文

2. 现在只能预测未来一天的。改成预测未来三天的，然后在图表中显示出来。

3. 网页加一个侧边栏，里面存着目前已经缓存好的预测数据。

   1. 侧边栏再加一个输入框。用户可以输入想要的公司名。server接收到这个公司名，然后现场下载、处理、预测新的dataset，将缓存加在侧边栏里。如果没有找到对应的公司名，则提示用户添加失败
   2. project_files文件夹中，再划分子文件夹，每个公司的文件存到单独的文件夹中
   3. 如果用户在聊天框里问了缓存中不存在的公司，则提示用户请在侧边栏中添加
   4. 将目前的“更新数据”按钮加在侧边栏的每一个缓存旁边。点击后更新对应的数据。

4. 网页改一下措辞，改成“Hello, I’m the Oracle. How can I help you today?”

   1. 用户会输入一个句子，里面包含两个元素：公司名、日期（日期可能以“tomorrow”这样的格式出现）

   2. 检测用户的两个输入元素：公司名、日期。然后显示出预测结果。在图表中，显示完整的过去30天以及未来3天，用折线连起来，高亮用户想问的那一天。

      1. 如果用户没有输入公司名，提示输入错误
      2. 如果用户没有输入日期，正常显示，但不高亮

   3. 在聊天栏里，说出那一天的公司股价是多少

   4. 下面是一个示例
      ```
      $ Hello, I’m the Oracle. How can I help you today?
      $ Can you give me the maximum temperature for tomorrow?
      $ Tomorrow’s maximum temperature will be 5 degrees Celsius
      ```

      
