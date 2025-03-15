1. “update all data”放在侧边框里
2. 我在尝试添加缓存时，输入了"NVIDIA"，提示“Failed to add symbol: Error adding symbol NVIDIA: 'NoneType' object has no attribute 'update'”，如何解决
3. 图表部分，把高亮功能去掉。
4. 图表类型，添加一个K线图。
    1. 当用户没有提及数据类型（例如最高价，最低价），默认显示K线图
    2. 对于其他图表，将历史数据的最后一天与预测数据的第一天用折线连接，使得图表更加美观
    3. 图表处，添加一个放大按钮，点击后可以全窗口查看图表

1. K线图显示不出来，并且提示“Sorry, an error occurred while processing your request. Please try again.”。请修复。
2. 只保留一个图表的下拉选择框，选项分别是candlestick，close，high，low，open，volume

1. 移除K线图
2. 侧边栏加一个刷新按钮，点击后，根据目前已有的文件夹名，确定已有的缓存

1. 我还是无法添加新的股票数据，例如我输入nvidia，客户端的终端会提示404 Client Error: Not Found for url: https://query2.finance.yahoo.com/v10/finance/quoteSummary/NVIDIA?modules=financialData%2CquoteType%2CdefaultKeyStatistics%2CassetProfile%2CsummaryDetail&corsDomain=finance.yahoo.com&formatted=false&symbol=NVIDIA&crumb=OQFEZmNMHCj
是否可以用其他方式下载股票数据，例如类似data_preparation.py中的方法

我发现用户尝试添加nvda后，只是下载了npy格式的数据，并没有预测后的数据，如何解决

删掉“通过输入添加新的股票数据”的功能