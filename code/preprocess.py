import pandas as pd
import numpy as np
import time
import itertools

#associations and interactions matrix
def get_dis_mi_adjacency_matrix(dis_mi_link):
    matrix = np.zeros((len(mi), len(dis)), dtype=np.float32)
    for index in dis_mi_link.index:
        matrix[dis_mi_link.loc[index, 'miRNA']][dis_mi_link.loc[index, 'disease']] = 1
    return matrix


def get_lnc_mi_adjacency_matrix(lnc_mi_link):
    matrix = np.zeros((len(lnc), len(mi)), dtype=np.float32)
    for index in lnc_mi_link.index:
        matrix[lnc_mi_link.loc[index, 'lncRNA']][lnc_mi_link.loc[index, 'miRNA']] = 1
    return matrix


def get_dis_lnc_adjacency_matrix(dis_lnc_link):
    matrix = np.zeros((len(lnc), len(dis)), dtype=np.float32)
    for index in dis_lnc_link.index:
        matrix[dis_lnc_link.loc[index, 'lncRNA']][dis_lnc_link.loc[index, 'disease']] = 1
    return matrix


#similarity

def compute_GIP(matrix, gamma=1):
    revised_gamma = matrix.shape[1] * gamma / np.sum(np.square(matrix))

    GIP_matrix = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=np.float32)
    for idx_1 in range(matrix.shape[1]):
        for idx_2 in range(matrix.shape[1]):
            print(f'computing GIP {idx_2} / {matrix.shape[1]} for {idx_1} / {matrix.shape[1]}', end='\r')
            GIP_matrix[idx_1][idx_2] = np.exp(- revised_gamma * np.sum(np.square(matrix[:, idx_1] - matrix[:, idx_2])))
    return GIP_matrix


def max_score(id, set):
    global gene_id, gene_sim, m_gene_id
    max_lls = 0.0
    for g in set:
        if g >= 0 and gene_sim[id][g] > max_lls:
            max_lls = gene_sim[id][g]
    return max_lls


def compute_func_sim(i, j, t):
    global genes, miRNA, miRNA1, m_gene_id
    if i == j:
        return 1.0
    elif miRNA1[i] not in miRNA or miRNA1[j] not in miRNA:
        return 0.0
    else:
        m1_gene = m_gene_id[i]
        m2_gene = m_gene_id[j]
        m1_gene_score = 0.0
        m2_gene_score = 0.0
        for k in range(len(m1_gene)):
            if m1_gene[k] in m2_gene:
                m1_gene_score += 1.0
            elif m1_gene[k] >= 0:
                m1_gene_score += max_score(m1_gene[k], m2_gene)
        for k in range(len(m2_gene)):
            if m2_gene[k] in m1_gene:
                m2_gene_score += 1.0
            elif m2_gene[k] >= 0:
                m2_gene_score += max_score(m2_gene[k], m1_gene)
        return (m1_gene_score + m2_gene_score) / (len(m1_gene) + len(m2_gene))

    
def semantic_value(theta, code, trs):
    code_split = code.split('.')
    code_len = len(code_split)

    code_value = 1.0
    for index in range(code_len, 0, -1):
        tmp_code = '.'.join(code_split[: index])
        if tmp_code in trs and trs[tmp_code] >= code_value:
            pass
        else:
            trs[tmp_code] = code_value
        code_value = code_value * theta
        

def miRNA_similarity(miRNA_gene_link,gene_gene_link_score,miss_miRNAlist):
    gene_gene_link_score['score'] = (gene_gene_link_score['score'] - gene_gene_link_score['score'].min()) / (gene_gene_link_score['score'].max() - gene_gene_link_score['score'].min())

    genes = set(gene_gene_link_score['EntrezGeneID1'].tolist()) | set(gene_gene_link_score['EntrezGeneID2'].tolist())
    genes = list(genes)
    gene_id = {g: i for i, g in enumerate(genes)}
    
    #mti means miRNA-targetgene-id  links from 
    mti_genes = miRNA_gene_link['Target Gene (Entrez Gene ID)'].unique().tolist()
    mti_sp_genes = list(set(mti_genes) - set(genes))
    mti_sp_genes_id = {g: -i - 1 for i, g in enumerate(mti_sp_genes)}
    
    miRNA1 = miRNA_disease_link['miRNA'].unique().tolist()
    miRNA2 = miRNA_gene_link['miRNA'].unique().tolist()
    miRNA = list(set(miRNA1) & set(miRNA2))
    miss_miRNA = list(set(miRNA1) - set(miRNA2))
    miRNA_id = {m: i for i, m in enumerate(miRNA1)}
    
    gene_sim = np.zeros((len(genes), len(genes)), dtype=np.float64)
    for idx in gene_gene_link_score.index:
        gene1 = gene_gene_link_score.loc[idx, 'EntrezGeneID1']
        gene2 = gene_gene_link_score.loc[idx, 'EntrezGeneID2']
        gene_sim[gene_id[gene1]][gene_id[gene2]] = gene_gene_link_score.loc[idx, 'score']
    
    for idx in range(len(genes)):
        gene_sim[idx][idx] = 1
   
    m_gene_id = []
    for m in miRNA1:
        m_gene = miRNA_gene_link.values[miRNA_gene_link.values[:, 0] == m][:, 1]
        m_gene_id.append([])
        for g in m_gene:
            if g in genes:
                m_gene_id[-1].append(gene_id[g])
            else:
                m_gene_id[-1].append(mti_sp_genes_id[g])
    
    func_sim = np.zeros((len(miRNA1), len(miRNA1)), dtype=np.float32)
    t = time.time()
    for i in range(len(miRNA1)):
        for j in range(len(miRNA1)):
            func_sim[i, j] = func_sim[j, i] = compute_func_sim(i, j, t)
            
    gip_sim = compute_GIP(dis_mi_matrix)
    miss_list = miss_miRNA
    mi_sim = np.zeros((len(miRNA1), len(miRNA1)), dtype=np.float32)
    for idx_1 in range(len(miRNA1)):
        for idx_2 in range(idx_1, len(miRNA1)):
            if mi[idx_1] in miss_list or mi[idx_2] in miss_list:
                mi_sim[idx_1, idx_2] = mi_sim[idx_2, idx_1] = gip_sim[idx_1, idx_2]
            else:
                mi_sim[idx_1, idx_2] = mi_sim[idx_2, idx_1] = func_sim[idx_1, idx_2]

    mi_similarity = mi_sim
    return mi_similarity

     
def disease_similarity(disease_name):
    disease_id = {d: i for i, d in enumerate(disease_name)}
    # mtrees is the mesh tree structure from MeSH database
    mtrees = pd.DataFrame()
    mtrees[0] = mtrees[0].apply(lambda x: str(x))
    disease_values_total = {}
    disease_values_structure = {}
    for disease in disease_name:
        disease_trs = {}
        disease_codes = mtrees.values[mtrees.values[:, 0] == disease][:, 1]
        disease_values_total[disease] = 0.0
        disease_values_structure[disease] = {}
        for disease_code in disease_codes:
            semantic_value(0.5, disease_code, disease_trs)
        for code in disease_trs:
            dn = mtrees.values[mtrees.values[:, 1] == code][:, 0][0]
            if dn not in disease_values_structure[disease]:
                disease_values_structure[disease][dn] = disease_trs[code]
                disease_values_total[disease] += disease_trs[code]
    miss_list = []
    for d in disease_name:
        if disease_values_total[d] == 0:
            miss_list.append(d)
    #disease semantic
    dis_similarity = np.zeros((len(disease_name), len(disease_name)), dtype=np.float64)
    for d1 in disease_name:
        for d2 in disease_name:
            if d1 == d2:
                dis_similarity[disease_id[d1]][disease_id[d2]] = 1.0
            else:
                d1_set = [x for x in disease_values_structure[d1]]
                d2_set = [x for x in disease_values_structure[d2]]
                inter_set = set(d1_set) & set(d2_set)
                inter_set = list(inter_set)
                inter_sum = 0.0
                if d1 in miss_list or d2 in miss_list:
                    dis_similarity[disease_id[d1]][disease_id[d2]] = 0
                else:
                    if len(inter_set) > 0:
                        for dn in inter_set:
                            inter_sum = inter_sum + disease_values_structure[d1][dn] + disease_values_structure[d2][dn]
                    dis_similarity[disease_id[d1]][disease_id[d2]] = inter_sum / (disease_values_total[d1] + disease_values_total[d2])                    
    return dis_similarity


def lncRNA_similarity(lncRNA_expression):
    from sklearn.preprocessing import minmax_scale
    lnc_spearman = lncRNA_expression.corr('spearman')
    lnc_similarity = minmax_scale(lnc_spearman, feature_range=(0, 1))
    return lnc_similarity


if __name__ == "__main__":
    dis_mi_link = pd.read_excel('./Associations_miRNAs_diseases.xlsx')
    dis_lnc_link = pd.read_excel('./Associations_lncRNAs_diseases.xlsx')
    lnc_mi_link = pd.read_excel('./Interactions_lncRNAs_miRNAs.xlsx')

    dis = dis_mi_link['disease'].unique().tolist()
    mi = dis_mi_link['mi'].unique().tolist()
    lnc = dis_lnc_link['lnc'].unique().tolist()
    
    dis_mi_matrix = get_dis_mi_adjacency_matrix(dis_mi_link)
    dis_lnc_matrix = get_dis_lnc_adjacency_matrix(dis_lnc_link)
    lnc_mi_matrix = get_lnc_mi_adjacency_matrix(lnc_mi_link)
    
    unlabeled = []
    d = [i for i in range(dis)]
    m = [m for m in range(mi)]
    l = [l for l in range(lnc)]
    for item1 in itertools.product(m, d):
        if dis_mi_matrix[item1[0], item1[1]] == 0:
            unlabeled.append(item1)
    for item2 in itertools.product(l, d):
        if dis_lnc_matrix[item2[0], item2[1]] == 0:
            unlabeled.append(item2)     
    unlabeled = np.array(unlabeled) 
    
    mi_similarity = miRNA_similarity()
    dis_similarity = disease_similarity()
    lnc_similarity = lncRNA_similarity()

    sim_m = np.concatenate([mi_similarity, dis_mi_matrix, lnc_mi_matrix.T], axis=1)
    sim_d = np.concatenate([dis_mi_matrix.T, dis_similarity,dis_lnc_matrix.T], axis=1)
    sim_l = np.concatenate([lnc_mi_matrix, dis_lnc_matrix, lnc_similarity], axis=1)
    sim_all = np.concatenate([sim_m, sim_d, sim_l], axis=0)
    #np.save("all_feature_adj", sim_all)