# 更新日志

*更新内容位于最上方。*

**4.22**

进度：

1. 开始编写卡口检测类
2. 对梯度累加曲线做平滑，使用标准差选取初始极值点
3. NMS方法精筛峰值点

**4.21**

关于卡口检测的想法：

1. 阀门圆形区域极坐标映射至平面坐标系
2. 水平方向梯度检测
3. 根据饱和度确定接口存在区域的mask，从而确定梯度图上对应区域
4. 垂直方向梯度值累加，统计方法寻找峰值
5. 使用半周长间距判断卡口峰值坐标

问题：

1. 对梯度累加曲线的增强方法？

**4.13**

关于卡口检测的想法：

1. 量化，例如以均分成8个扇形区域，确定所在区域即可
2. 梯度，使用梯度信息检测
3. 对称，使用对称思想减少运算

**4.12**

1. 完成RANSAC椭圆拟合
2. 更新ValveDetector函数说明
3. 更新SwitchDetector，SwitchMatcher函数说明

**4.8**

完善了接口中心寻找思路：

1. 接口ROI降采样
2. 使用canny边缘而非二值图进行椭圆拟合

**4.6**

关于接口ROI细化寻找中心的思路：

1. 预处理，得到接口的前景区域，二值边缘图，由形态学计算得来
2. 边缘筛选，根据大小去掉太小的区域
3. 对剩下的点，借助RANSANC思想，引入随机的椭圆拟合，具体的椭圆拟合算法先用传统的

**4.4**

完成：

1. 修改`keras-yolo3/yolo_video.py`，进行批量测试
2. 使用216张测试图片进行测试，全部能检测到接口
3. 编写可视化训练过程脚本

继续要做的：同4.3剩余内容。

**4.3**

之前SSD512训练失败，似乎是没有收敛，但是训练过程中损失很低，数量级位于1e-6。
使用YOLOv3训练了收敛模型，目前测试效果不错，能完成ROI检测。

接下来：

1. 完成测试集`2007_test.txt`中全部数据的测试，统计准确率
2. 转移服务器训练、测试相关数据至本地
3. 设计合适的水泵中心定位算法
4. YOLOv3对于该任务场景的优化（不知道能不能做成）
5. 对整个设计进行对比试验数据/算法选择，准备对比试验

**4.1**

使用SSD512原始结构，无预训练权重，从头开始训练对水泵接口定位的网络，目前训练效果不错。

训练平台：

- Ubuntu 16.04
- 单块GTX1080，CUDA 9.0，cuDNN 7.0，tensorflow-gpu 1.8.0

可以优化的地方：

1. 单目标检测，环境变化不大，考虑将网络结构简化，以SSD300作为原始模型
2. （非任务相关）远程训练监视，Keras回调RemoteMonitor使用，参考http://vra.github.io/2018/03/18/keras-callbacks-remote-monitor/
3. 迁移学习应该是更好的选择，不过按照测试，在没有pre-trained model情况下直接从头训练效果亦可；找到模型再考虑

**3.23**

研究了基于模板匹配的方法，大致分为：

1. MAD算法，$d(x,y)=\sum_i^M \sum_j^N |S(x+i,y+j) - T(i,j)|$，类似于目前采用的方法，对噪声敏感
2. NCC算法，使用相关系数作为判据，计算代价大
3. SSDA算法，引入随机性计算累计误差，比MAD快但准确率低，对噪声敏感
4. PFC算法，使用二进制分块编码，对光照高度敏感

现在使用NCC算法。

问题：

1. 获取ROI候选区的流程能否优化
2. 运算量能否优化

**3.20**

针对使用基于学习的分类器的开关筛选，进行文献查阅，确定了可以采用的特征：

1. 区域均值
2. 区域方差
3. 平滑度，由方差定义
4. 区域熵
5. 三阶矩
6. 主色调直方图
7. 纹理特征
8. 投影直方图

**3.19**

整理一下之前多日的更新。

更新：

1. 阀门图像收集757张，并完成标注
2. 开关识别测试脚本
3. 简单location registration脚本
4. `utils.py`中加入批量降采样
5. 开关测试图像64张，准确率60/64

问题：

1. 开关ROI筛选机制需要优化，重点研究一下，除了模板匹配还能用什么方法
2. SwitchDetector使用的宽高比阈值和误差精度阈值需要实验确定，做二维grid search

**3.14**

更新：

1. 开关检测测试程序
2. 更新一些函数说明
3. 开关检测正确率测试，29/32，正确率0.90625

**3.13**

问题：

1. 结合投票数与对称性进行接口筛选

**3.11**

更新：

1. PCA方法给开关区域降维可能不适用，因为没有大量数据，无统计意义

问题：

1. 水泵接口检测加入候选区筛选机制，先找到圆形区域，然后选出正确区域
2. 缩放因子的确定：$\alpha = \frac{2r_{max}}{\min\{width, height\}}$，参数：$r_{max}$
3. 缩放因子使用：$\alpha \times (width, height)$
4. 对开关区域用方差/熵进行筛选
5. 开关区域打分函数设计

**3.9**

更新：

1. 使用模板匹配方法筛选开关候选区

问题：

1. 模板匹配方法有提升空间，可考虑PCA降维表示特征再计算损失

**3.8**

更新：

1. 完成Hough变换不同尺度检测数量测试
2. 处理前进行高斯平滑
3. 对候选区域轮廓面积进行过滤，除去太小的轮廓区域
4. 调整更合适的宽高比误差阈值为0.6
5. 调整更合适的轮廓面积筛选阈值为50

**3.7**

更新：

1. 只检测正面即可（接近正圆），原因：之前角度修正已经正对箱体
2. 使用S通道分割水泵接口，二值化、高斯模糊、双边滤波
3. 缩放图像进行多级Hough变换，二分法确定最佳缩放比例；依据：检测数量与尺寸缩放比例的关系，用多组测试展示正相关性（放服务器上跑）

问题：

1. 合适的水泵接口建模方法
2. 合适的卡槽检测方法

**2.28**

更新：

1. 修复ROI遮罩创建时错位的bug
2. 加入bounding box大小筛选

问题：

1. 用于bounding box筛选的参数是否可以通过统计确定？

**2.27**

问题：

1. 直接使用ORB特征点/描述子对新测试图像进行修正，发现特征点太少，难以计算透视变换矩阵
2. 对开关红色的识别失败，原因是使用OpenCV的H通道下red range有间断，原先方法无法直接提取连续区间

更新：

1. 为角度修正功能增加新解决方案：FAST特征点+SIFT描述子，解决特征点不足的问题，需要测试更好的角点/描述子组合
2. 将R通道与B通道交换，然后转换至HSV空间，此时原先的red range改变为连续的blue range，可以直接提取

存在的问题：

1. FAST角点使用阈值区分，对噪声高度敏感，进行平滑处理的参数需要微调
2. 可能有更好的角点/描述子组合方式，可进行实验，自变量：角点种类，描述子种类，不同角度的消防箱图像；评价标准：运行速度，角度修正效果
3. 初步试验用FAST角点+SIFT特征，修正效果更好，时间开销较小，后续实验可以记录量化指标
4. 用作registration的正面照在只包含消防箱金属框以内部分的情况下，尽可能大
5. 需要对候选ROI进行筛选，可以基于宽长比初筛

**2.26**

采集了新的消防箱测试照片。

**2.24**

导师反馈意见：

1. 沟通使用doc文档
2. 光照处理建议
   1. 增加光照，而非除去光照，光照波长需试验确定
   2. 对图像进行移动差分处理，类似$I_d(x,y) = I(x+u,y+v)-I(x,y)$
   3. 对灰度图像逐像素非线性映射至更易识别的数据点位

**2.18**

1. 初步实现从消防箱正面图中找到、筛选开关ROI

ROI定位可能存在的问题：

1. 相似色调噪声，需要一种更精细的筛选方式确定候选区

**2.11**

1. 先检测形状的想法效果不好，改为先色彩聚类，然后分簇拟合
2. 新想法，在registration中记录开关ROI大致位置，透视变换后能直接确定大致位置
3. 重新调整了流程如下（对于开关定位）：
   1. 角度校正
   2. 光照校正：gamma强化hue红色区，抑制其他区
   3. 应用registration：正面照片中定义ROI区域，指出开关所在区域
   4. 初筛：高斯平滑，二值化，确定候选区
   5. fine-tuning：筛选匹配候选区，定位

**2.10**

1. 实现了同态滤波，效果一般
2. 尝试对HSV的V进行直方图均衡+中值滤波，亮度调整明显
3. Retinex光照增强暂时不加入，参考：https://www.ixueshu.com/document/15b7e2dc1bc5c37c318947a18e7f9386.html
4. 调整事前registration策略：拍摄正面照+记录开关位置（左右）
5. 开始实现基于形状角的方法，形状角值与文献不太相同？

**2.9**

1. 光照补偿尝试同态滤波/HDR merging，HDR需要多张不同曝光时间图像
2. 同态滤波实现参考：https://www.ixueshu.com/document/5dfc6a081c5158da318947a18e7f9386.html

**12.23**

开题中要解决哪些问题，寻找哪些文献？

1. 光照如何补偿？
2. 拍摄角度如何补偿？
3. 基于哪种识别？具体实现参考？

**12.18**

数据收集，消防箱与水泵：

- 不同光照
- 不同角度
- 不同地点的设备

实现想法：

1. 实现层次化检测：较强低通滤波/降采样（可以加速？），分割，迭代进行，逐渐降低滤波强度至确定开关附近的金属框ROI，最后一个层次下用颜色定位
2. 寻找水泵：颜色+网格特征抽取分类，确定初步ROI；寻找卡口角点，圆形接口拟合

**12.14**

一些想法：

- 基于分割的方法
  - 基于聚类
    - K-means
    - Meanshift
  - 基于图
    - 最大流/最小割划分
  - 基于特征值分析
    - 图→权重矩阵→特征值分解
- 基于形态学的方法
  - 边缘检测
  - 形状拟合

**11.15**

- 算法综述？写需要的算法以及最后选用的？
  - 聚类+分类的方法，定位，按性能做一下对比，有图表
  - 先找个好理解的应用一下，后期再考虑更高级算法
  - 找开源工具
- 做成什么样子，最后产出一个系统化工具？半自动or自动？
  - 确定一个成果构想：算法结果即可
- 论文里面写什么，怎么写？
  - 绪论：意义，研究现状（简单提到即可），章节安排
  - 算法分析：算法具体实现
  - 实验：实现算法，对比效果，分章节写，描述整个过程
  - 总结展望