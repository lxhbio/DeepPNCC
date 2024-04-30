#!/usr/bin/env python
"""
TODO:
# Author: 
# Created Time : 

# File Name: 
# Description: 

"""


import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='TODO')

    # IO and norm options #输入数据的路径设置
    parser.add_argument('--mk_model', '-mdl', type=str, help='TODO: Input model folder path')  # 输入存储模型的文件夹的路径设置
    parser.add_argument('--mk_output', '-opt', type=str, help='TODO: Input output folder path')  # 输入存储输出文档的文件夹的路径设置

    parser.add_argument('--exp', '-e', type=str, help='TODO: Input gene expression data path')  #输入表达数据的路径设置
    parser.add_argument('--adj', '-a', type=str, help='Input adjacency matrix data path')    #输入邻接矩阵的路径设置
    parser.add_argument('--coordinate', '-c', type=str, help='Input cell coordinate data path')    ###输入坐标文件的路径设置#####
    parser.add_argument('--reference', '-r', type=str, help='Input cell type label path')  #输入细胞类型数据的路径设置
    parser.add_argument('--verbose', action='store_true', help='Print loss of training process')
    parser.add_argument('--outdir', '-o', type=str, default='output/', help='Output path')  #输出路径
    parser.add_argument('--outpostfix', '-n', type=str, help='The postfix of the output file')  #输出文件的后缀
    parser.add_argument('--log_add_number', '-add', type=float, default=None, help='Perform log10(x+log_add_number) transform')  #执行log10（x+log_add_number）转换'
    parser.add_argument('--fil_gene', type=int, default=None, help='Remove genes expressed in less than fil_gene cells')  #删除在少于fil_gene细胞中表达的基因
    parser.add_argument('--latent_feature', '-l', default=None, help='')

    # Training options
    parser.add_argument('--test_ratio', '-t', type=int, default=0.1, help='Testing set ratio (>1: edge number, <1: edge ratio; default: 0.1)')   #测试集比率（>1：边缘数，<1：边缘比率；默认值：0.1）
    parser.add_argument('--iteration', '-i', type=int, default=5, help='Iteration (default: 40)')  #迭代（默认值：5）
    parser.add_argument('--encode_dim', '-edim', type=int, nargs=2, default=[125, 125], help='Encoder structure')     #编码器结构
    parser.add_argument('--regularization_dim', '-rdim', type=int, nargs=2, default=[150, 125], help='Adversarial regularization structure') #对抗性规则化结构  #TODO:[125, 125, ]
    parser.add_argument('--lr1', type=float, default=0.0004, help='TODO')
    parser.add_argument('--lr2', type=float, default=0.0008, help='TODO')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight for L2 loss on latent features')
    parser.add_argument('--dropout', '-d', type=float, default=0.5, help='Dropout rate (1 - keep probability)')  #dropout率
    parser.add_argument('--features', type=int, default=1, help='Whether to use features (1) or not (0)')
    parser.add_argument('--seed', type=int, default=7, help='Random seed for repeat results') #TODO:50,7,1
    parser.add_argument('--activation', type=str, default='relu', help="Activation function of hidden units (default: relu)")  #激活函数选择（默认：relu）
    parser.add_argument('--init', type=str, default='glorot_uniform', help="Initialization method for weights (default: glorot_uniform)")  #权重的初始化方法（默认值：glorot_uniform）
    parser.add_argument('--optimizer', type=str, default='Adam', help="Optimization method (default: Adam)")  #优化方法（默认值：Adam）

    # Clustering options
    parser.add_argument('--cluster', action='store_true', help='TODO')
    parser.add_argument('--cluster_num', type=int, default=None, help='TODO')    

    parser.add_argument('--gpu', '-g', type=int, default=0, help='Select gpu device number for training') #选择用于培训的gpu设备编号
    # tf.test.is_built_with_cuda()

    # parser.set_defaults(transpose=False,
    #                     testsplit=False,
    #                     saveweights=False,
    #                     sizefactors=True,
    #                     batchnorm=True,
    #                     checkcounts=True,
    #                     norminput=True,
    #                     hyper=False,
    #                     debug=False,
    #                     tensorboard=False,
    #                     loginput=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Select gpu device number  
    import os 
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  #if you use GPU, you must be sure that there is at least one GPU available in your device
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #set only using cpu

    # Import modules
    try:
        import tensorflow as tf  # import tf and the rest module after parse_args() to make argparse help show faster
    except ImportError:
        raise ImportError('DeepPNCC requires TensorFlow. Please follow instructions'
                          ' at https://www.tensorflow.org/install/ to install'
                          ' it.')
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.spatial.distance import pdist, squareform
    import copy
    from deeppncc.io import *
    from deeppncc.plot import *
    from deeppncc.utils import sparse2tuple, packed_data, set_placeholder, set_optimizer, update, ranked_partial
    from deeppncc.models import Deeppncc, Discriminator
    from deeppncc.metrics import linkpred_metrics, select_optimal_threshold
    from deeppncc.enrichment import connection_number_between_groups, generate_adj_new_long_edges, edges_enrichment_evaluation
    from deeppncc.sensitivity import get_sensitivity
    from deeppncc.cluster import clustering

    # Set random seed
    seed = args.seed
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Import and pack datasets
    exp_df, adj_df = read_dataset(args.exp, args.adj, args.fil_gene, args.log_add_number) #args.exp--表达矩阵  args.adj--邻接矩阵  其余为参数
    write_csv_matrix(exp_df, './exp_df')
    write_csv_matrix(adj_df, './adj_df')

    exp, adj = exp_df.values, adj_df.values

    print(exp_df.isnull().any().sum())#检查数据中是否有缺失值，输出为 “0”， 说明不含空值
    print(np.isnan(exp_df).any().sum())
    print(np.isinf(exp_df).all().sum())#检测数据中是否有无穷数据，

    print(adj_df.isnull().any().sum())
    print(np.isnan(adj_df).any().sum())
    print(np.isinf(exp_df).all().sum())

    # coord_df = read_coordinate(args.coordinate)  ######     args.coordinate--坐标文件    ##############################################################
    # coord = coord_df.values
    # cell_label_df = read_cell_label(args.reference)  #args.reference--细胞类型注释文件
    # cell_label = cell_label_df.values

    feas = packed_data(exp, adj, args.test_ratio)
    var_placeholders = set_placeholder(feas['adj_train'], args.encode_dim[1])

    # Output some basic information
    cell_num = exp.shape[0]
    gene_num = exp.shape[1]
    predefined_edge_num = np.where(adj==1)[0].shape[0]/2   #预定义 局部连接边数
    print('predefined_edge_num:', predefined_edge_num)

    print("\n**************************************************************************************************************")
    print("  DeepPNCC: De novo reconstruction of cell interaction landscapes from single-cell spatial transcriptome data  ")
    print("**************************************************************************************************************s\n")
    print("======== Parameters ========")
    print('Cell number: {}\nGene number: {}\nPredefined local connection number: {}\niteration: {}'.format(
            cell_num, gene_num, predefined_edge_num, args.iteration))
    print("============================")

    print(os.getcwd())
    # Create storage folders  #创建存储文件夹
    os.mkdir(args.mk_model)
    os.mkdir(args.mk_output)

    # os.mkdir("./model")
    # os.mkdir("./output")

    # Building model and optimizer
    # dims = []
    deeppncc = Deeppncc(var_placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'], args.encode_dim[0], args.encode_dim[1])
    deeppncc_discriminator = Discriminator(args.encode_dim[1], args.regularization_dim[0], args.regularization_dim[1])  #鉴别器
    opt = set_optimizer(deeppncc, deeppncc_discriminator, var_placeholders, feas['pos_weight'], feas['norm'], feas['num_nodes'], args.lr1, args.lr2)  #优化器

################################################################################################################
    # Fitting model  #拟合模型
    # Saver 将训练好的模型参数保存起来，以便以后进行验证或测试，这是我们经常要做的事情。tf里面提供模型保存的是tf.train.Saver()模块。
    saver = tf.train.Saver(max_to_keep=1)  #max_to_keep 参数，这个是用来设置保存模型的个数，默认为5，即 max_to_keep=5，保存最近的5个模型。只想保存最后一代的模型，则只需要将max_to_keep设置为1即可

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Metrics list
    train_loss = []
    test_ap = []
    path = args.mk_output
    # latent_feature = None
    max_test_ap_score = 0

    #我加的  创建train_record.csv用于记录训练记录
    df = pd.DataFrame(columns=["Epoch", "train_loss", "test_roc", "test_ap", "test_acc"])
    df.to_csv(args.mk_output + '/train_record.csv', index=False)


    # Train model
    for epoch in range(args.iteration):

        emb_hidden1_train, emb_hidden2_train, avg_cost_train = update(deeppncc, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], var_placeholders, feas['adj_train'], args.dropout, args.encode_dim[1])
        # print("emb_hidden1_train", emb_hidden1_train)
        # print("emb_hidden2_train", emb_hidden2_train)
        # print(np.isinf(emb_hidden2_train).all().sum())
        # print("train_loss", avg_cost_train)



        train_loss.append(avg_cost_train)

        lm_train = linkpred_metrics(feas['test_edges'], feas['test_edges_false'])

        roc_score, ap_score, acc_score, _ = lm_train.get_roc_score(emb_hidden2_train, feas)
        test_ap.append(ap_score)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost_train), "test_roc=", "{:.5f}".format(roc_score), "test_ap=", "{:.5f}".format(ap_score))

        ##
        list = ['%04d' % (epoch + 1), "{:.5f}".format(avg_cost_train), "{:.5f}".format(roc_score), "{:.5f}".format(ap_score), "{:.5f}".format(acc_score)]
        record = pd.DataFrame([list])
        record.to_csv(args.mk_output + '/train_record.csv', mode='a', header=False, index=False)

        if ap_score > max_test_ap_score:  #判断输出 精度最高 那一代的模型
            max_test_ap_score = ap_score

            # saver.save(sess, './model/'+str(args.outpostfix), global_step=epoch+1)  #修改之前saver.save(sess, './model/'+args.outpostfix, global_step=epoch+1)
            saver.save(sess, args.mk_model+'/'+str(args.outpostfix), global_step=epoch+1)  #修改之前saver.save(sess, './model/'+args.outpostfix, global_step=epoch+1)

            np.save(args.mk_output+'/emb_hidden1_'+str(epoch+1)+'.npy', emb_hidden1_train)
            np.save(args.mk_output+'/emb_hidden2_'+str(epoch+1)+'.npy', emb_hidden2_train)

            latent_feature = copy.deepcopy(emb_hidden2_train)

            write_json({"Epoch:": '%04d' % (epoch + 1), "train_loss=": "{:.5f}".format(avg_cost_train),
                        "test_roc=": "{:.5f}".format(roc_score), "test_ap=": "{:.5f}".format(ap_score),
                        "args.dropout:": args.dropout, "args.encode_dim:": args.encode_dim, "args.regularization_dim:": args.regularization_dim},
                       args.mk_output + '/train_loss_' + str(args.outpostfix))  # 优化后的阈值 最大准确率得分

    plot_evaluating_metrics(test_ap, path, "epoch", "score", ["AUPRC"], "AUPRC")
    write_pickle(feas, args.mk_output+'/feas')


################################################################################################################
    print("args.dropout:", args.dropout)



    ### Output ###
    # 3.1. 概率、连接和阈值
    adj_reconstructed_prob, adj_reconstructed, _, _, all_acc_score, max_acc_score, optimal_threshold = select_optimal_threshold(feas['test_edges'], feas['test_edges_false']).select(latent_feature, feas)
    print("optimal_threshold:", optimal_threshold)
    print("max_acc_score:", max_acc_score)

    write_json(all_acc_score, args.mk_output+'/acc_diff_threshold_'+str(args.outpostfix))  #所有准确率得分
    write_json({'optimal_threshold':optimal_threshold, 'max_acc_score':max_acc_score}, args.mk_output+'/threshold_'+str(args.outpostfix))  #优化后的阈值 最大准确率得分
    write_csv_matrix(adj_reconstructed_prob, args.mk_output+'/adj_reconstructed_prob_'+str(args.outpostfix))  ###### 重构的 邻接矩阵概率表 ###########################################################
    write_csv_matrix(adj_reconstructed, args.mk_output+'/adj_reconstructed_'+str(args.outpostfix))  #重构的 邻接矩阵表
    print("Eed!!!")

    print("args.encode_dim:", args.encode_dim[0], args.encode_dim[1])
    print("args.regularization_dim:", args.regularization_dim[0], args.regularization_dim[1])

    print("args.encode_dim:", args.encode_dim)
    print("args.regularization_dim:", args.regularization_dim)






    # # 3.2. 距离分布、距离矩阵
    # adj_diff = adj - adj_reconstructed
    # write_csv_matrix(adj_diff, './output/adj_diff_raw')
    # adj_diff = (adj_diff == -1).astype('int')  #预测出来的邻接关系 ---> -1变1，已有的或之前没有的 ---> 0  变成新的[0,1]矩阵
    # write_csv_matrix(adj_diff, './output/adj_diff')
    # adj_diff = sp.csr_matrix(adj_diff)   #将值与行列索引的数组一起存储

    #
    # dist_matrix_rongyu = pdist(coord, 'euclidean')   #pdist()    输出为一个列表  每个坐标反复对比求距离--这里计算的是‘euclidean’（欧几里得距离）
    # dist_matrix = squareform(dist_matrix_rongyu)     #squareform()    将样本间距离用方阵表示出来。
    # write_csv_matrix(dist_matrix, './output/dist_matrix')
    #
    # #这里好像直接adj_diff乘上dist_matrix就可以
    # new_edges = sparse2tuple(sp.triu(sp.csr_matrix(adj_diff)))[0]     #sp.triu()功能：取出稀疏矩阵的上三角部分的非零元素
    # all_new_edges_dist = dist_matrix[new_edges[:, 0].tolist(), new_edges[:, 1].tolist()]
    # plot_histogram(all_new_edges_dist, xlabel='distance', ylabel='density', filename='all_new_edges_distance', color="coral")
    # write_csv_matrix(dist_matrix*adj_diff, './output/all_new_edges_dist_matrix')
    #
    # # 3.3. 连接可视化
    # id_subgraph, _ = ranked_partial(adj, adj_reconstructed, coord, [10,15])  #返回的是[(diff,[id_list]),(diff,[id_list])...]这种形式  #size是list，[3,5]代表把总图切成宽3份(x)、高5份(y)的子图
    #                                                                                             #adj_rec1:[10,15], adj_rec2:[3,5]
    # rank = 0
    # for item in id_subgraph:
    #     cell_type_subgraph = cell_label[item[1],:][:,[0,1]]
    #     cell_type_subgraph[:,0] = np.array(list(range(cell_type_subgraph.shape[0]))) + 1  #需要对X重新生成细胞的id，这里以1开始
    #     coord_subgraph = coord[item[1],:]
    #     adj_reconstructed_subgraph = adj_reconstructed[item[1],:][:,item[1]]
    #     rank += 1
    #     adjacency_visualization(cell_type_subgraph, coord_subgraph, adj_reconstructed_subgraph, filename='spatial_network_rank'+str(rank)+'_diff'+str('%.3f'%item[0]))
    #
    # # 4. 互作强度
    # cutoff_distance = np.percentile(all_new_edges_dist,99)  #求取all_new_edges_dist数列第99%分位的数值
    # print("cutoff_distance:", cutoff_distance)
    #
    # connection_number, _ = connection_number_between_groups(adj, cell_label[:,1])
    # write_csv_matrix(connection_number, './output/connection_number_between_groups')  #细胞类型之间的连接数目矩阵
    #
    # adj_new_long_edges = generate_adj_new_long_edges(dist_matrix, new_edges, all_new_edges_dist, cutoff_distance)
    # write_csv_matrix(adj_new_long_edges.todense(), './output/adj_new_long_edges')
    #
    # print('------permutations calculating------')
    # cell_type_name = [np.unique(cell_label[cell_label[:,1]==i,2])[0] for i in np.unique(cell_label[:,1])]
    # test_result, _, _, _ = edges_enrichment_evaluation(adj, cell_label[:,1], cell_type_name, edge_type='all edges')
    # write_csv_matrix(test_result, './output/all_edges_enrichment_evaluation', colnames=['cell type A','cell type B','average_connectivity','significance'])
    # test_result, _, _, _ = edges_enrichment_evaluation(adj_new_long_edges.toarray(), cell_label[:,1], cell_type_name, edge_type='long edges', dist_matrix=dist_matrix, cutoff_distance=cutoff_distance)
    # write_csv_matrix(test_result, './output/long_edges_enrichment_evaluation', colnames=['cell type A','cell type B','connection_number','significance'])
    #
    # # 5. 敏感性
    # # get_sensitivity(exp_df, feas, './model/'+args.outpostfix)
    #
    # # 6. 聚类
    # if args.cluster:
    #     cluster_num = args.cluster_num
    #     cluster_label = clustering(latent_feature, cluster_num)
    #     write_csv_matrix(cluster_label, './output/label', colnames=['cell_id','cluster_id'])







