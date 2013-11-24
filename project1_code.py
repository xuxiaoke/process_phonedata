############################ the first project#############################
###########################By Xiaoke 2013.11.22#############################
############################################################################

###########################set the work path#############################
##import OS
##os.chdir("/Users/xiaokeeie/all_datasets/SM_datasets")

#########################import library#####################################
from scipy import sparse
import networkx as nx
import random
###########################end#########################################

#######################CN function######################################
def CN(G,edge_list):
    """
    common_neighors(G, edge_list)   
    calculate the common neighors of two nodes in edge_list    
    Parameters
    ----------  
    G:            - networkx graph 
    edge_list     - pairs of nodes for sum the common neighbors    
    """
    common_number=[]
    for item in edge_list:
        tmp_count=0
        if (item[0] in G) and (item[1] in G):
            n1=G.neighbors(item[0])
            n2=G.neighbors(item[1])
            if len(n1)>0 and len(n2)>0:
                for i1 in n1:
                    if i1 in n2:
                        tmp_count=tmp_count+1
        
        common_number.append(tmp_count)
    return common_number       
###########################end#########################################

#######################WCN function######################################
def WCN(G,edge_list):
    """
    common_neighors(G, edge_list)    
    calculate the weighted common neighors of two nodes in edge_list   
    Parameters
    ----------  
    G:            - networkx graph 
    edge_list     - pairs of nodes for sum the common neighbors    
    """
    wcommon_number=[]
    for item in edge_list:
        tmp_count=0
        if (item[0] in G) and (item[1] in G):
            n1=G.neighbors(item[0])
            n2=G.neighbors(item[1])
            if len(n1)>0 and len(n2)>0:
                for i1 in n1:
                    if i1 in n2:
                        tmp_count=tmp_count+G[item[0]][i1]['weight']+G[item[1]][i1]['weight']
        
        wcommon_number.append(tmp_count/2)
    return wcommon_number       
###########################end#########################################

#######################AUC function######################################
def AUC(real_edges,false_edges):
    """
    AUC(input1,input2)   
    the statistic for comparing two different link prediction methods   
    Parameters
    ----------  
    input1:       - the result of the first test dataset: real existing links
    input2:       - the result of the second test dataset: no existing links    
    """
    AUC_result=0
    for i in range(0,len(real_edges)):
        if real_edges[i]>false_edges[i]:
            AUC_result=AUC_result+1
        elif real_edges[i]==false_edges[i]:
            AUC_result=AUC_result+0.5
            
    return AUC_result/len(real_edges)      
###########################end#########################################

###########################read data#########################################
myfile=open("SD03.txt",'r')
origin_data=myfile.readlines()
myfile.close()
###########################end#########################################

#######################change string into int format########################
smd=[]
for item in origin_data:
    data_tmp=item.split()
    smd.append([int(data_tmp[0])-1,int(data_tmp[1])-1])
###########################end#########################################

###########################find the max size#########################################
matrix_size=0
for item in smd:
    tmp_size=max(item)
    if tmp_size>matrix_size:
        matrix_size=tmp_size
###########################end#########################################

###########################sparse matrix #########################################
data_matrix=sparse.dok_matrix((matrix_size+1,matrix_size+1))
for item in smd:
    if data_matrix[item[0],item[1]]>0:
              data_matrix[item[0],item[1]]=data_matrix[item[0],item[1]]+1
    else:
              data_matrix[item[0],item[1]]=1
###########################end#########################################

###########################reciprocial process##############################
for item in smd:
    if data_matrix[item[0],item[1]]>0 and data_matrix[item[1],item[0]]>0:
        data_matrix[item[0],item[1]]=(data_matrix[item[0],item[1]]+data_matrix[item[1],item[0]])/2
        data_matrix[item[1],item[0]]=data_matrix[item[0],item[1]]
    elif data_matrix[item[0],item[1]]>0 or data_matrix[item[1],item[0]]>0:
        data_matrix[item[0],item[1]]=0  
        data_matrix[item[1],item[0]]=0
###########################end#########################################
        
#####################remove no useful data##################################
for i in range(0,matrix_size+1):
    data_matrix[i,i]=0
###########################end#########################################

###########################network to list ()weight edges###################    
##generate the full network by sparse matrix
all_graph=nx.from_scipy_sparse_matrix(data_matrix)

## from network to list  
smd1=[]
for i,j,weight_data in all_graph.edges(data=True):
    if 'weight' in weight_data:
           smd1.append([i,j,weight_data['weight']])
###########################end#########################################

##################rescale this data, generate new list####################             
node_dict={}    
i=0
for item in smd1:
    try:
        node_dict[item[0]]
    except:
        node_dict[item[0]]=i
        i=i+1
    try:
        node_dict[item[1]]
    except:
        node_dict[item[1]]=i
        i=i+1
 
new_list=[]
for item in smd1:
    new_list.append([node_dict[item[0]],node_dict[item[1]],item[2]])
################################end####################################  

##################generate test data1 and train data#######################
train_list=new_list
test_list=[]

test_ratio=0.1
test_length=int(test_ratio*len(new_list))
for item in range(0,test_length):
    line=random.choice(train_list)
    test_list.append(line)
    train_list.remove(line)
###########################end#########################################


##################generate test data2####################################
no_length=0
no_list=[]

while no_length<test_length:
    index_1=random.randint(0,len(new_list)-1)
    index_2=random.randint(0,len(new_list)-1)
    if data_matrix[index_1,index_2]==0 and index_1!=index_2:
        no_list.append([min(index_1,index_2),max(index_1,index_2)])
        no_length=no_length+1
###########################end#########################################


###generate the train network
train_graph=nx.Graph()
train_graph.add_weighted_edges_from(train_list)

###calculating CN
CN_test1=CN(train_graph,test_list)
CN_test2=CN(train_graph,no_list)

###calculating WCN
WCN_test1=WCN(train_graph,test_list)
WCN_test2=WCN(train_graph,no_list)

###calculating AUC
AUC_CN=AUC(CN_test1,CN_test2)
AUC_WCN=AUC(WCN_test1,WCN_test2)

###show the results
print(AUC_CN)
print(AUC_WCN)



