# dsci-6000p

**UPDATE 12.11** \
在baseline的基础上修改得到了new_baseline，代码更方便大家进行修改，另外我添加了时间特征以及部分时序数据特征，大家可以在如下代码中添加新的特征，如注释所示。目前由于某种原因，结果好像还不如原来的baseline 😭 大家可以再试试
```python
# 特征工程
def make_features(data):
    data = data.drop(['train or test'],axis=1)
    community_list = []

    
    for flow in tqdm(range(20)):
        # 添加初始特征
        features_list = []  
        features_list.append(pd.DataFrame(data.loc[:,[f'flow_{flow+1}']]))
        
        cls_feature = pd.DataFrame(np.zeros((data.shape[0],5)), columns=[f'cls_{i}' for i in range(5)])
        cls_list = [f'cls_{i}' for i in range(5) if (int(flow / (2**i)) % 2 == 1)]
        cls_feature.loc[:, cls_list] = 1
        features_list.append(cls_feature)

        # 未处理特征 n*2: time | flow_i
        orign_features = pd.DataFrame(data.loc[:,['time',f'flow_{flow+1}']])
    
        '''
        在此添加特征，用函数process_xxx_features封装，将输出结果添加到features_list, 例如:

        features_list.append(process_xxx_features(orign_features, f'flow_{flow+1}'))

        origin_features 为一个n*2的DataFrame，包含time和flow_i列。
        f'flow_{flow+1}'是对应小区流量的列名称。
        return 一个 除去 origin_features的列的新DataFrame
        '''
        # 添加时间特征
        features_list.append(process_time_features(orign_features, f'flow_{flow+1}'))

        # 添加数据特征
        features_list.append(process_data_features(orign_features, f'flow_{flow+1}'))





        # n, c
        new_features = np.concatenate(features_list, axis=-1)
        
        community_list.append(new_features)

    # n*d*c
    new_data = np.stack(community_list,axis=0).transpose(1,0,2)

    return new_data

```