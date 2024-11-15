import subprocess
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import openpyxl
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.font_manager import FontProperties
import pickle
import os
import matplotlib as mpl
# 设置 Matplotlib 的默认字体
mpl.font_manager.fontManager.addfont('Arial Unicode MS.ttf')
mpl.font_manager.fontManager.addfont('Apple Color Emoji.ttf')
plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'Apple Color Emoji'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False#用来正常显示负号
st.title('团体明细网络图生成器')
group_id = int(st.text_input('请输入要查询的团体id', '2523'))
## 缓存数据
@st.cache_data
def load_data():
    try:
        author_info_df = pd.read_feather('作者明细-包含团体id.fth')
        data_use_df = pd.read_feather('作者网络明细回查表11.2.fth')

    except FileNotFoundError as e:
        st.error(f"文件未找到: {e}")
        st.stop()
    return author_info_df, data_use_df

author_info, data_use = load_data()
@st.cache_data
def load_graph():
    with open('average_cnt.gpickle','rb') as f:
        G_all = pickle.load(f)
    with open('live_cnt.gpickle','rb') as f:
        G_live_cnt = pickle.load(f)
    with open('comment_cnt.gpickle','rb') as f:
        G_comment_cnt = pickle.load(f)
    with open('live_play_cnt.gpickle','rb') as f:
        G_live_play_cnt = pickle.load(f)
    with open('send_message_cnt.gpickle','rb') as f:
        G_send_message_cnt = pickle.load(f)
    with open('co_relation_num.gpickle','rb') as f:
        G_co_relation_num = pickle.load(f)
    with open('comments_at_author.gpickle','rb') as f:
        G_comments_at_author = pickle.load(f)
    with open('common_hard_fans_cnt.gpickle','rb') as f:
        G_common_hard_fans_cnt = pickle.load(f)
    return G_all, G_live_cnt, G_comment_cnt, G_live_play_cnt, G_send_message_cnt, G_co_relation_num, G_comments_at_author, G_common_hard_fans_cnt
G_all, G_live_cnt, G_comment_cnt, G_live_play_cnt, G_send_message_cnt, G_co_relation_num, G_comments_at_author, G_common_hard_fans_cnt = load_graph()
# 获取节点数据，根据用户输入的团体id
node_df = author_info[author_info['group'] == group_id][['作者id', '作者昵称']]
# 绘制局部网络图
def plot_local_group_graph(G, node_df, title, edge_width_scale=1.0, figsize=(15, 10)):
    node_ids = list(node_df['作者id'])
    node_dict = dict(zip(node_df['作者id'], node_df['作者昵称']))
    session_data = author_info[author_info['作者id'].isin(node_df['作者id'].tolist())][['作者id','30d日均23-总打开理由']]
    session_dict = dict(zip(session_data['作者id'],session_data['30d日均23-总打开理由']))
    for node_id, node_name in node_dict.items():
        G.nodes[node_id]['name'] = node_name
    for node_id, node_value in session_dict.items():
        G.nodes[node_id]['value'] = node_value

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111)
    subgraph = G.subgraph(node_ids)
    edge_weights = [subgraph[u][v]['weight'] for u, v in subgraph.edges()]
    edge_widths = [w * edge_width_scale for w in edge_weights]
    weights = [subgraph[u][v]['weight'] for u, v in subgraph.edges()]

    # 归一化权重
    norm = Normalize(vmin=min(weights), vmax=max(weights))
    cmap = plt.cm.viridis  # 选择颜色映射
    mappable = ScalarMappable(norm=norm, cmap=cmap)

    # 颜色映射
    edge_colors = [mappable.to_rgba(w) for w in weights]
    edge_widths = [w * 0.2 for w in weights]  # 调整线宽
    pos = nx.spring_layout(subgraph, k = 5) ## k的大小用来调节节点之间的散布状况。
    node_sizes = [subgraph.nodes[node]['value'] * 0.1 for node in subgraph.nodes()]
    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color='skyblue', ax=ax)
    nx.draw_networkx_edges(subgraph, pos, width=edge_widths, alpha=0.7, edge_color=edge_colors, ax=ax)
    # 标签绘制，更改为节点大小为打开理由绝对值规模
    labels = {node: G.nodes[node]['name'] for node in subgraph.nodes()}
    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=10, font_family='sans-serif', ax=ax)
    edge_labels = {(u, v): f"{subgraph[u][v]['weight']:.2f}" for u, v in subgraph.edges()}  # 保留两位小数
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8, font_family='sans-serif', ax=ax)
    ax.set_title(title, fontsize=24)
    ax.axis('off')

    ax.patch.set_facecolor('lightgray')  # 设置背景色
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)

    plt.rcParams['font.sans-serif']=['Arial Unicode MS', 'Apple Color Emoji'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False#用来正常显示负号
    plt.colorbar(mappable, ax=ax, label='边权重大小')
    st.pyplot(fig)

# 图表选项
graph_options = {
    '综合指标关系网': G_all,
    '直播互动关系网': G_live_cnt,
    '视频评论关系网': G_comment_cnt,
    '直播互相观看关系网': G_live_play_cnt,
    '私信关系网': G_send_message_cnt,
    '共创&作品艾特关系网': G_co_relation_num,
    '用户相互艾特作者关系网': G_comments_at_author,
    '共同铁粉关系网': G_common_hard_fans_cnt
}

# 选择绘制图形，在options当中存储字典
selected_graph = st.selectbox('请选择要绘制的关系网', list(graph_options.keys()))

# 生成关系网络图按钮
if st.button('生成关系网络图'):
    st.write(f'Generating chart for group ID: {group_id} and graph: {selected_graph}')
    plot_local_group_graph(graph_options[selected_graph], node_df, selected_graph, edge_width_scale=0.2)
def data_info(group_id,selected_graph, data_use, author_info):
    node_df = author_info[author_info['group'] == group_id][['作者id']]
    temp_data = data_use[(data_use['t1.source_user_id'].isin(node_df['作者id'].tolist()))&(data_use['t1.target_user_id'].isin(node_df['作者id'].tolist()))]
    if selected_graph == '综合指标关系网':
        temp_data = temp_data[['t1.source_user_id','t1.target_user_id','source_author_name','target_author_name',
                               'source_author_fans_user_num','target_author_fans_user_num','average_cnt']]
        temp_data.rename(columns={'t1.source_user_id':'作者id_1','t1.target_user_id':'作者id_2','source_author_name':'作者1昵称',
                                 'target_author_name':'作者2昵称','source_author_fans_user_num':'作者1粉丝量',
                                  'target_author_fans_user_num':'作者2粉丝量','average_cnt':'综合指标互动次数'}, inplace=True)
    elif selected_graph == '直播互动关系网':
        temp_data = temp_data[['t1.source_user_id','t1.target_user_id','source_author_name','target_author_name',
                               'source_author_fans_user_num','target_author_fans_user_num','live_cnt',
                              'live_cnt_contribute']]
        temp_data.rename(columns={'t1.source_user_id':'作者id_1','t1.target_user_id':'作者id_2','source_author_name':'作者1昵称',
                                 'target_author_name':'作者2昵称','source_author_fans_user_num':'作者1粉丝量',
                                  'target_author_fans_user_num':'作者2粉丝量','live_cnt':'直播互动次数',
                                  'live_cnt_contribute':'直播互动贡献度'}, inplace=True)
    elif selected_graph == '视频评论关系网':
        temp_data = temp_data[['t1.source_user_id','t1.target_user_id','source_author_name','target_author_name',
                               'source_author_fans_user_num','target_author_fans_user_num','comment_cnt',
                              'comment_cnt_contribute']]
        temp_data.rename(columns={'t1.source_user_id':'作者id_1','t1.target_user_id':'作者id_2','source_author_name':'作者1昵称',
                                 'target_author_name':'作者2昵称','source_author_fans_user_num':'作者1粉丝量',
                                  'target_author_fans_user_num':'作者2粉丝量','comment_cnt':'视频相互评论次数',
                                 'comment_cnt_contribute':'视频互动贡献度'}, inplace=True)
    elif selected_graph == '直播互相观看关系网':
        temp_data = temp_data[['t1.source_user_id','t1.target_user_id','source_author_name','target_author_name',
                               'source_author_fans_user_num','target_author_fans_user_num','live_play_cnt',
                               'live_play_cnt_contribute']]
        temp_data.rename(columns={'t1.source_user_id':'作者id_1','t1.target_user_id':'作者id_2','source_author_name':'作者1昵称',
                                 'target_author_name':'作者2昵称','source_author_fans_user_num':'作者1粉丝量',
                                  'target_author_fans_user_num':'作者2粉丝量','live_play_cnt':'直播互相观看次数',
                                 'live_play_cnt_contribute':'直播观看贡献度'}, inplace=True)
    elif selected_graph == '私信关系网':
        temp_data = temp_data[['t1.source_user_id','t1.target_user_id','source_author_name','target_author_name',
                               'source_author_fans_user_num','target_author_fans_user_num','send_message_cnt',
                              'send_message_cnt_contribute']]
        temp_data.rename(columns={'t1.source_user_id':'作者id_1','t1.target_user_id':'作者id_2','source_author_name':'作者1昵称',
                                 'target_author_name':'作者2昵称','source_author_fans_user_num':'作者1粉丝量',
                                  'target_author_fans_user_num':'作者2粉丝量','send_message_cnt':'私信互动数',
                                 'send_message_cnt_contribute':'私信互动贡献度'}, inplace=True)
    elif selected_graph == '共创&作品艾特关系网':
        temp_data = temp_data[['t1.source_user_id','t1.target_user_id','source_author_name','target_author_name',
                               'source_author_fans_user_num','target_author_fans_user_num','co_relation_num',
                              'co_relation_contribute']]
        temp_data.rename(columns={'t1.source_user_id':'作者id_1','t1.target_user_id':'作者id_2','source_author_name':'作者1昵称',
                                 'target_author_name':'作者2昵称','source_author_fans_user_num':'作者1粉丝量',
                                  'target_author_fans_user_num':'作者2粉丝量','co_relation_num':'共创&标题艾特数',
                                 'co_relation_contribute':'作品共创&标题艾特贡献度'}, inplace=True)
    elif selected_graph == '用户相互艾特作者关系网':
        temp_data = temp_data[['t1.source_user_id','t1.target_user_id','source_author_name','target_author_name',
                               'source_author_fans_user_num','target_author_fans_user_num','comments_at_author']]
        temp_data.rename(columns={'t1.source_user_id':'作者id_1','t1.target_user_id':'作者id_2','source_author_name':'作者1昵称',
                                 'target_author_name':'作者2昵称','source_author_fans_user_num':'作者1粉丝量',
                                  'target_author_fans_user_num':'作者2粉丝量','comments_at_author':'用户相互艾特数'}, inplace=True)
    elif selected_graph == '共同铁粉关系网':
        temp_data = temp_data[['t1.source_user_id','t1.target_user_id','source_author_name','target_author_name',
                               'source_author_fans_user_num','target_author_fans_user_num','common_hard_fans_cnt']]
        temp_data.rename(columns={'t1.source_user_id':'作者id_1','t1.target_user_id':'作者id_2','source_author_name':'作者1昵称',
                                 'target_author_name':'作者2昵称','source_author_fans_user_num':'作者1粉丝量',
                                  'target_author_fans_user_num':'作者2粉丝量','common_hard_fans_cnt':'共同铁粉数'}, inplace=True)
    return temp_data 
temp_data = data_info(group_id, selected_graph, data_use, author_info)
temp_data
