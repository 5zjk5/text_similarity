import pandas as pd
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import HeatMap


corrs = pd.read_csv('../output/fuzzywuzzy.csv',encoding='gbk',index_col=0)

# 保存 xy 轴标签和长度
x = y = list(corrs.index)
l = len(x)

# 把相关系数 DataFrame 转换为列表
corrs = np.array(corrs)
corrs = corrs.tolist()

# 对应热力图的值
value = [[i,j,corrs[i][j]] for i in range(l) for j in range(l)]

c = (
    HeatMap()
    .add_xaxis(x)
    .add_yaxis("", y, value)
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-60)),
        title_opts=opts.TitleOpts(title=""),
        visualmap_opts=opts.VisualMapOpts(is_show=False,min_=0,
                                          max_=100),
    )
)
c.render('../output/热力图.html')