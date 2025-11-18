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

