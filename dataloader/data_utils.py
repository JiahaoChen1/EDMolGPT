#!/usr/bin/env python
# coding=utf-8
import os
import pickle
from rdkit import Chem


def convert_to_file(pickle_path, out_dir, max_rows = -1, file_list=None):
    with open(pickle_path, "rb") as f:
        rows_list = pickle.load(f)
    if max_rows > 0:
        max_num = min(len(rows_list),max_rows)
    else:
        max_num = len(rows_list)
    acc_num = 0
    for row in rows_list:
        if acc_num >= max_num: break

        mol_name = row["mol_smiles"].replace('/','_')
        with open(os.path.join(out_dir, f"{mol_name}.sdf"),"w") as f:
            f.write(row["mol_conf_sdf"])
        for idx, resolution in enumerate(row["mol_ed_res"]):
            with open(os.path.join(out_dir, f"{mol_name}_res{resolution}.pdb"), "w") as f:
                f.write(row["mol_ed_pdb"][idx])
        file_list.append(mol_name + '\n')

        acc_num += 1
                
def ed_pdb_read(ed_pdb_path):
    mol = Chem.MolFromPDBFile(ed_pdb_path, sanitize=False, removeHs=False)
    if mol is None:
        print("错误：无法解析 PDB 文件")
        return
    # 获取第一个构象的坐标（PDB通常只有一个构象）
    conf = mol.GetConformer()
    # 遍历所有原子
    cnt = 0
    elements = []
    poss = []
    for atom in mol.GetAtoms():
        # 获取原子类型（元素符号）
        element = atom.GetSymbol()
        
        # 获取坐标
        pos = conf.GetAtomPosition(atom.GetIdx())
        x, y, z = pos.x, pos.y, pos.z
        cnt += 1
        elements.append(element)
        poss.append([x, y, z])
    return elements, poss
    #     print(f"原子类型：{element}，坐标：({x:.3f}, {y:.3f}, {z:.3f})")
    # print(cnt)

def read_point_cloud_gen(path):
    res = []
    with open(path) as f:
        for line in f.readlines():
            res.append(line.strip())
    
    elements = []
    poss = []
    for i in range(len(res)):
        if i >= 2:
            pc_type, pc_x, pc_y, pc_z = res[i].split(' ')
            poss.append([float(pc_x), float(pc_y), float(pc_z)])
            elements.append(pc_type)
    # print(len(poss), len(elements))
    return elements, poss



def convert_pocket():
    import numpy as np
    import os
    import sys
    sys.path.append('/home/jiahao_chen/code/Lingo3DMol-main/')
    from util.pocket_code_all import PocketCode
    from tqdm import tqdm

    # data_root_dir = os.getcwd() + '/superpdb_v1_update_nci_ed/'
    data_root_dir='/disk/jiahao_chen/rna_task/superpdb_v1_update_nci_ed/'
    os.makedirs('/home/jiahao_chen/code/Lingo3DMol-main/datasets/pocket/pocket_pdb',exist_ok=True)
    os.makedirs('/home/jiahao_chen/code/Lingo3DMol-main/datasets/pocket/mol_sdf',exist_ok=True)
    os.makedirs('/home/jiahao_chen/code/Lingo3DMol-main/datasets/pocket/ed',exist_ok=True)

    for data_file in tqdm(os.listdir(data_root_dir)):
        for data in np.load( data_root_dir + data_file , allow_pickle=True):
            pocket_pdb = data['pocket_pdb']
            pdb_id = data['pdb_id']
            mol_sdf = data['mol_conf_sdf']
            mol_ed_pdbs = data['mol_ed_pdb']
            mol_ed_ress = data['mol_ed_res']
            print(pocket_pdb, file=open('/home/jiahao_chen/code/Lingo3DMol-main/datasets/pocket/pocket_pdb/'+pdb_id+'.pdb','w'))
            print(mol_sdf, file=open('/home/jiahao_chen/code/Lingo3DMol-main/datasets/pocket/mol_sdf/'+pdb_id+'_ligand.sdf','w'))
            for ed_pdb, ed_res in zip(mol_ed_pdbs,mol_ed_ress):
                print(ed_pdb,file=open('/home/jiahao_chen/code/Lingo3DMol-main/datasets/pocket/ed/'+pdb_id+f'_res{ed_res}.pdb','w'))

    pdb_root_dir = '/home/jiahao_chen/code/Lingo3DMol-main/datasets/pocket/pocket_pdb/'

    for pocket_pdb_file in os.listdir(pdb_root_dir):
        final_symbol, final_reses, mask,final_pos, center,contact, contact_scaffold = PocketCode().pocketCodeNCI(pdb_root_dir + pocket_pdb_file)
        print(final_pos)
        raise ValueError


def read_dude(pickle_path, out_dir):
    with open(pickle_path, "rb") as f:
        rows_list = pickle.load(f)

    cnt = 0
    correspondance = {}
    for row in rows_list:
        if not ('_nh' in row):
            cnt += 1
            continue
        file = rows_list[row]
        mol_name = file["mol_smiles"].replace('/','_')
        correspondance[mol_name] = row
        continue
        
        with open(os.path.join(out_dir, f"{mol_name}.sdf"),"w") as f:
            f.write(file["mol_conf_sdf"])
        
        for idx, resolution in enumerate(file["mol_ed_res"]):
            with open(os.path.join(out_dir, f"{mol_name}_res{resolution}.pdb"), "w") as f:
                f.write(file["mol_ed_pdb"][idx])
    
    print(correspondance)
    print(cnt)

if __name__ == '__main__':
    file_list = []
    # for i in range(591,592):
    #     p = f'/disk/jiahao_chen/edmol/pretrain_592pretrain_592.csv_split1_ed/ed_mol_chunk{i}.pkl'
    #     convert_to_file(p, "/home/jiahao_chen/code/Lingo3DMol-main/datasets/valset_large", max_rows=-1, file_list=file_list)
    #     print(i)
    # convert_pocket()
    # read_point_cloud_gen('/home/jiahao_chen/code/Lingo3DMol-main/pocket-ligand/case_30/ed.xyz')
    read_dude('/home/jiahao_chen/code/Lingo3DMol-main/datasets/DUDE_ed.pkl', out_dir='/home/jiahao_chen/code/Lingo3DMol-main/datasets/dude_pc')