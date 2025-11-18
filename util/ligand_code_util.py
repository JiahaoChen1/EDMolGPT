#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import traceback
import warnings
from copy import deepcopy

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdGeometry

import torch

from .find_root import find_root_smi
# from smiles_encoding import atomwise_tokenizer

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings(action="ignore")

def degrees(x):
    return x * (180.0 / torch.pi)

def atomwise_tokenizer(smi):
    """
    Tokenize a SMILES molecule at atom-level:
        (1) 'Br' and 'Cl' etc. are two-character tokens
        (2) Symbols with bracket are considered as tokens
    """
    import re
    pattern = r"(\(\[[*]\]\)|\[[*]\]|He|Li|Be|Ne|Na|Mg|Al|Si|Cl|Ar|Ca|Ti|Cr|Mn|Fe|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|Rb|Sr|Zr|Mo|Tc|Ru|Rh|Pd|Ag|Cd|Te|Xe|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Er|Tm|Lu|Hf|Ta|Re|Ir|Pt|Au|Hg|Tl|Bi|At|Rn|Fr|Ra|Ac|Th|Pa|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|Lr|Rf|Db|Sg|Bh|Mt|Ds|Rg|Nh|Fl|Mc|Lv|Ts|Og|H|B|C|N|O|F|P|S|K|V|Y|I|U|b|c|n|o|s|p|\[|\]|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@{2}|@|\|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    
    return tokens

class LigandCodeUtil:
    atom_vocab_list = [
        "<PAD>", "<SOS>", "<EOS>", "<SEP>", 
        "C", "N", "O", "S", "P", 
        "F", "Cl", "Br", "I", 
        "c", "n", "o", "s",
        "UnkElement", "UnkAtomType", "UnkSymbol", "H",
        "+", "-", 
        "/", "\\", "@", "@@", "#", "=", 
        "1", "2", "3", "4", "5", "6", 
        "(", ")", "[", "]", "[*]", "([*])",
        "<S2OS>", "<S3OS>", "<MASK>", "<FILL>", "<FEOS>",
    ]
    atom_vocab_i2c = {i: x for i, x in enumerate(atom_vocab_list)}
    atom_vocab_c2i = {x: i for i, x in enumerate(atom_vocab_list)}

    ring_vocab_list = [
        '0',
        '3', '3+3', '3+4', '3+5', '3+5+5', '3+5+6', '3+6', '3+6+6', '3+7',
        '4', '4+4', '4+4+4', '4+4+5', '4+5', '4+5+5', '4+5+6', '4+6', '4+6+6', '4+7', '4+7+7', '4+8',
        '5', '5+5', '5+5+5', '5+5+6', '5+5+7', '5+6', '5+6+6', '5+6+7', '5+6+8', '5+7', '5+7+7', '5+8',
        '6', '6+6', '6+6+6', '6+6+7', '6+6+8', '6+7', '6+7+7', '6+7+8', '6+8', '6+8+8',
        '7', '7+7', '7+8',
        '8', '8+8', '8+8+8'
    ]
    ring_vocab_i2c = {i: x for i, x in enumerate(ring_vocab_list)}
    ring_vocab_c2i = {x: i for i, x in enumerate(ring_vocab_list)}

    fsmiles_vocab_list = [
        "<PAD>", "<SOS>", "<EOS>", "<SEP>",
        "C_0", "C_3", "C_3+3", "C_3+4", 
        "C_3+5", "C_3+5+5", "C_3+5+6", 
        "C_3+6", "C_3+6+6", "C_3+7", 
        "C_4", "C_4+4", "C_4+4+4", "C_4+4+5", 
        "C_4+5", "C_4+5+5", "C_4+5+6", 
        "C_4+6", "C_4+6+6", "C_4+7", "C_4+7+7", "C_4+8",
        "C_5", "C_5+5", "C_5+5+5", "C_5+5+6", "C_5+5+7", 
        "C_5+6", "C_5+6+6", "C_5+6+7", "C_5+6+8", "C_5+7", "C_5+8", 
        "C_6", "C_6+6", "C_6+6+6", "C_6+6+7",
        "C_6+7", "C_6+7+7", "C_6+7+8",
        "C_6+8", "C_6+8+8", 
        "C_7", "C_7+7", "C_7+8", 
        "C_8", "C_8+8", "C_8+8+8",
        "N_0", "N_3", "N_4", "N_4+5", "N_4+6", "N_4+7", "N_4+8",
        "N_5", "N_5+5", "N_5+5+5", "N_5+5+6", 
        "N_5+6", "N_5+6+6", "N_5+7", "N_5+8",
        "N_6", "N_6+6", "N_6+6+6", "N_6+6+7",
        "N_6+7", "N_6+7+7", "N_6+8",
        "N_7", "N_7+7", "N_8",
        "O_0", "O_3", "O_4", "O_4+6",
        "O_5", "O_5+5", "O_5+6", "O_5+7",
        "O_6", "O_6+6", "O_6+7", 
        "O_7", "O_7+7", "O_8",
        "S_0", "S_3", "S_4", "S_5", "S_5+5", "S_5+6",
        "S_6", "S_6+6", "S_7", "S_8",
        "P_0", "P_5", "P_6", "P_7",
        "F_0", "Cl_0", "Br_0", "I_0",
        "c_0", "c_3", "c_4", "c_4+6",
        "c_5", "c_5+5", "c_5+5+6", "c_5+6", "c_5+6+6", "c_5+6+7", "c_5+6+8",
        "c_5+7", "c_5+7+7", "c_5+8",
        "c_6", "c_6+6", "c_6+6+6", "c_6+6+7", "c_6+6+8", "c_6+8+8",
        "c_6+7", "c_6+7+7", "c_6+8", "c_7", "c_7+7",
        "n_0", "n_5", "n_5+5", "n_5+5+6",
        "n_5+6", "n_5+6+6", "n_5+6+7", "n_5+7", "n_5+7+7", "n_5+8",
        "n_6", "n_6+6", "n_6+6+6",
        "n_6+7", "n_6+7+7", "n_6+8", "n_7",
        "o_5", "o_6", "s_5", "s_6",
        "UnkElement", "UnkAtomType", "UnkSymbol", "H", #155, 156, 157,
        "<S2OS>", "<S3OS>", "<MASK>", "<FILL>", "<FEOS>",
        "Undefined1", "Undefined2", "Undefined3", "Undefined4", "Undefined5",
        "Undefined6", "Undefined7", "Undefined8", "Undefined9", "Undefined10",
        "Undefined11", "Undefined12", "Undefined13", "Undefined14", "Undefined15",
        "Undefined16", "Undefined17","Undefined18", "Undefined19", "Undefined20",
        "Undefined21", "Undefined23", "Undefined24", "Undefined25", "Undefined26",
        "Undefined27", "Undefined28", "Undefined29", "Undefined30", "Undefined31",
        "Undefined32", "Undefined33", "Undefined34", "Undefined35", "Undefined36",
        "Undefined37", "Undefined38", "Undefined39", "Undefined40", "Undefined41",
        "Undefined42", "Undefined43", "Undefined44", "Undefined45", "Undefined46",
        "Undefined47", "Undefined48", "Undefined49", "Undefined50", "Undefined51",
        "Undefined52", "Undefined53", "Undefined54", "Undefined55", "Undefined56",
        "Undefined57", "Undefined58", "Undefined59", "Undefined60", "Undefined61",
        "Undefined62", "Undefined63", "Undefined64", "Undefined65", "Undefined66",
        "Undefined67", "ced_res0", "ced_res1", "ced_res2", "ced_res3",
        "ced_res4", "ced_res5",
        "+", "-", "/", "\\", "@", "@@", "#", "=",
        "1", "2", "3", "4", "5", "6",
        "(", ")", "[", "]", "[*]", "([*])"
    ]
    fsmiles_vocab_i2c = {i: x for i, x in enumerate(fsmiles_vocab_list)}
    fsmiles_vocab_c2i = {x: i for i, x in enumerate(fsmiles_vocab_list)}

    ele_tokens = list(range(4, 19))

    def __init__(self, scale=24, resolution=0.1, max_token_length=100):
        self.bond_length_grid = {
            0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8:0, 9:0,
            10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 15: 6,
            16: 7, 17: 8, 18: 9, 19: 10, 20: 11, 21: 12
        }
        # 离散化box的尺寸
        self.scale = scale
        self.resolution = resolution
        self.grid_size = int(scale / resolution)
        self.unk_pos = self.grid_size
        self.max_token_length = max_token_length

        self.inversed_map = {0:0, 1:10, 2:11, 3:12, 4:13, 5:14, 6:15, 7:16, 8:17, 9:18, 10:19, 11:20, 12:21}

    def mol_with_atom_index(self, mol):
        for atom in mol.GetAtoms():
            atom.SetProp("atomNum", str(atom.GetIdx()))

        return mol

    def tree_decomp(
        self,
        mol,
        reserve_ids=None,
        reserve_only_seed_flag=True,
        keep_bonds_prob=1.0,
        dummy_coord=None,
        merge_small_frags=True,
    ):
        """
        把分子给切除
        """
        atoms_ids = {}
        # check_frag = []
        for i in mol.GetAtoms():
            i.SetIntProp("atom_idx", i.GetIdx())
            # atoms_ids[i.GetIdx()] = i.GetAtomicNum()

        for i in mol.GetBonds():
            i.SetIntProp("bond_idx", i.GetIdx())

        # print(atoms_ids)
        cut_bonds = []
        cut_bonds_conn = {}
        
        for bond in mol.GetBonds():

            bond_type = bond.GetBondTypeAsDouble()
            bond_idx = bond.GetIdx()

            if bond.IsInRing():
                continue
            if bond_type != 1.0:
                continue

            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            ##和环连接的单独原子，不切割
            if (a1.IsInRing() and (not a2.IsInRing())):
                if a2.GetDegree() == 1:
                    continue

            if (a2.IsInRing() and (not a1.IsInRing())):
                if a1.GetDegree() == 1:
                    continue

            if a1.IsInRing() or a2.IsInRing():
                if a1.GetSymbol() == "H" or a2.GetSymbol() == "H":
                    continue
                cut_bonds.append(bond_idx)
                cut_bonds_conn[a1.GetIdx()] = bond_idx
                cut_bonds_conn[a2.GetIdx()] = bond_idx

        if reserve_ids is not None:
            cutable_bonds = []
            for bond_id in cut_bonds:
                bond = mol.GetBondWithIdx(bond_id)
                b1, b2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if (b1 in reserve_ids) and (b2 in reserve_ids):
                    continue
                cutable_bonds.append(bond_id)
            cut_bonds = cutable_bonds

        # cut_bonds保留的是符合fsmiles语法规则的片段的切除位点，此处引入了随机性，以一定概率丢掉某些切除位点，即使片段不分离
        if keep_bonds_prob != 1.0:
            keep_bonds = []
            for bond_id in cut_bonds:
                if np.random.random() < keep_bonds_prob:
                    keep_bonds.append(bond_id)
            cut_bonds = keep_bonds

        # if merge_small_frags:
        #     frags = Chem.FragmentOnBonds(mol, cut_bonds)
        #     try:
        #         frags = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=True)
        #     except Exception as e:
        #         frags = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=False)
        #     for idx, frag in enumerate(frags):
        #         num_heavy_atoms = frag.GetNumHeavyAtoms()
        #         if num_heavy_atoms < 3:


        if len(cut_bonds) > 0:
            frags = Chem.FragmentOnBonds(mol, cut_bonds)
            try:
                frags = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=True)
            except Exception as e:
                frags = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=False)
        else:
            frags = [mol]


        # 合并算法
        for frag in frags:
            frag = Chem.RWMol(frag)
            temp = []
            flag_ring = []
            for atom in frag.GetAtoms():
                if atom.GetSymbol() == "*":
                    continue
                else:
                    temp.append(int(atom.GetProp("atom_idx")))
                    flag_ring.append(atom.IsInRing())

            if len(temp) <= 3:
                for idx, flag in zip(temp, flag_ring):
                    if (idx in cut_bonds_conn) and flag:
                        bond_del = cut_bonds_conn[idx]
                        # print(cut_bonds, bond_del)
                        try:
                            cut_bonds.remove(bond_del)
                        except:
                            pass
                        break
        # print(cut_bonds)
        
        if len(cut_bonds) > 0:
            frags = Chem.FragmentOnBonds(mol, cut_bonds)
            try:
                frags = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=True)
            except Exception as e:
                frags = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=False)
        else:
            frags = [mol]

        cliques = []
        frag_list = []

        for frag in frags:
            # check_frag.append(frag)
            frag = Chem.RWMol(frag)
            temp = []
            for atom in frag.GetAtoms():
                if atom.GetSymbol() == "*":
                    continue
                else:
                    temp.append(int(atom.GetProp("atom_idx")))
            # print(temp)
            if dummy_coord is not None:
                is_dummy_frag = False
                for i in range(len(frag.GetAtoms())):
                    if frag.GetAtomWithIdx(i).GetSymbol() == "*":
                        continue
                    pos = frag.GetConformer().GetAtomPosition(i)
                    if (pos.x == dummy_coord.x) and (pos.y == dummy_coord.y) and (pos.z == dummy_coord.z):
                        is_dummy_frag = True
                        break

                if is_dummy_frag:
                    continue

            if (reserve_ids is not None) and reserve_only_seed_flag and (0 not in temp):
                continue
            elif (reserve_ids is not None) and (not reserve_only_seed_flag) and (0 in temp):
                frag_list.insert(0, frag)
                cliques.insert(0, deepcopy(temp))
            else:
                frag_list.append(frag)
                cliques.append(deepcopy(temp))
        # print(cliques, frag_list)
        # assert 0
        # print(cut_bonds_conn)
        # for frag in frags:
        #     print(Chem.MolToSmiles(frag))
        # assert 0
        return cliques, frag_list, atoms_ids

    def get_clique_mol(self, frag, index, root_index, already=[], pos_linker=None, removeFirst=True):
        """
        :param frag: 目前要处理的片段 mol
        :param index:      该片段上的连接位点
        :param root_index: 该片段上的根位点
        :param already:    该片段上的idx
        :param ind:        总的来说 第几个片段
        :param removeFirst:
        :return:
        """

        # remove rooted * if exist (not first frag)
        if removeFirst:
            flag = False
            for i in range(frag.GetNumAtoms()):
                atom = frag.GetAtomWithIdx(i)
                idx = int(atom.GetProp("atomNum"))
                if atom.GetSymbol() != "*" and idx == index:  # 找到切割的原子
                    neighbors = atom.GetNeighbors()
                    delete = []
                    for n in neighbors:
                        if n.GetSymbol() == "*":
                            p = frag.GetConformer().GetAtomPosition(n.GetIdx())
                            if p.x == pos_linker[0] and p.y == pos_linker[1] and p.z == pos_linker[2]:
                                delete.append(n.GetIdx())
                                flag = True
                                break

                    delete = sorted(delete, reverse=True)  # TODO
                    for d in delete:
                        mw = Chem.RWMol(frag)
                        mw.RemoveAtom(d)
                        frag = mw.GetMol()
                    if flag:
                        break

        # rootedAt rooted_idx
        rooted_idx = -1
        for i in range(frag.GetNumAtoms()):
            atom = frag.GetAtomWithIdx(i)
            if atom.GetSymbol() != "*":
                idx = int(atom.GetProp("atomNum"))
                if idx == index:
                    rooted_idx = atom.GetIdx()
                    break

        Chem.MolToSmiles(frag, rootedAtAtom=rooted_idx)
        r_order1 = frag.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"]
        frag = Chem.RenumberAtoms(frag, r_order1)

        # final smiles
        frag_smiles = Chem.MolToSmiles(frag, rootedAtAtom=0)
        frag_smiles, rorder_next = self.parse_star(frag_smiles)

        # next neighbor
        ori_next = []
        for i in range(frag.GetNumAtoms()):
            atom = frag.GetAtomWithIdx(i)
            symbol = atom.GetSymbol()
            if symbol == "*":
                atom_neighbors = atom.GetNeighbors()
                for neighbor in atom_neighbors:
                    ori_idx = int(neighbor.GetProp("atomNum"))
                    if ori_idx in already:
                        ori_next.append(ori_idx)

        frag_next = [[a1, a2] for a1, a2 in zip(ori_next, rorder_next)]

        position = []
        for i in range(0, frag.GetNumAtoms()):
            symbol = frag.GetAtomWithIdx(i).GetSymbol()
            if symbol != "*":
                pos = frag.GetConformer().GetAtomPosition(i)
                position.append([pos.x, pos.y, pos.z])

        neighbors = []
        frag_indices = []
        for i in range(0, frag.GetNumAtoms()):
            atom = frag.GetAtomWithIdx(i)
            atom_neighbors = atom.GetNeighbors()
            atom_idx = atom.GetIdx()
            neighbor_indices = []
            symbol = atom.GetSymbol()
            if symbol == "*":
                continue
            frag_indices.append(int(atom.GetProp("atomNum")))

            if atom_idx == 0:
                neighbors.append(root_index)
                continue
            for n in atom_neighbors:
                n_idx = n.GetIdx()
                if n_idx < atom_idx:
                    neighbor_indices.append(int(n.GetProp("atomNum")))

            neighbor_indices = sorted(neighbor_indices)
            neighbors.append(neighbor_indices[-1])
        # frag_smiles: 返回 连接位点 为 0 的frag
        # frag_next: root idx 和 star 对应的 global idx
        # 重原子对应的坐标
        # 邻居
        # 当前原子的 idx
        # 当前的片段 reorder 根据 smiles 顺序
        return frag_smiles, frag_next, position, neighbors, frag_indices, frag

    def parse_star(self, smi):
        label = []
        """
        栈结构，倒序pop *
        """
        reversed_smi = smi[::-1]
        for i, s in enumerate(reversed_smi):
            if s == "*":
                for j in range(i, len(reversed_smi)):
                    if reversed_smi[j] == "[":
                        label.append([len(reversed_smi) - j - 1, len(reversed_smi) - i - 1])
                        break
        res = ""
        next = []
        if len(label) > 0:
            for pre, post in label:
                res = smi[: pre + 1] + smi[post:]
                next.append(int(smi[pre + 1 : post]))
                smi = res

            return res, next[::-1]
        else:
            return smi, []

    def flatten_seq(
        self,
        mol,
        initial_index=0,
        reserve_ids=None,
        reserve_only_seed_flag=True,
        keep_bonds_prob=1.0,
        dummy_coord=None,
    ):
        """
        给定分子，切出分子片段
        """
        mol = self.mol_with_atom_index(mol)
        cliques, frags, atoms_ids = self.tree_decomp(
            mol,
            reserve_ids,
            reserve_only_seed_flag,
            keep_bonds_prob,
            dummy_coord=dummy_coord,
        )
        # print(cliques, frags)
        smiles = []
        already = []
        pre_v = []
        next_stack = []
        removeFirst = False
        # 重原子坐标
        positions = []
        # 邻居
        neighbors = []
        ori_indices = []
        final_frags = []

        for frag_idx, (clique, frag) in enumerate(zip(cliques, frags)):
            if initial_index in clique:
                (
                    frag_smiles,
                    frag_next,
                    frag_coords,
                    frag_neighbors,
                    frag_ori_indices,
                    frag,
                ) = self.get_clique_mol(frag, initial_index, initial_index, clique, removeFirst=removeFirst)
                removeFirst = True
                smiles.append(frag_smiles)
                already.append(frag_idx)
                positions.extend(frag_coords)
                neighbors.extend(frag_neighbors)
                ori_indices.extend(frag_ori_indices)
                final_frags.append(frag)

                for n in frag_next:
                    if sorted(n) not in pre_v and sorted(n) not in next_stack:
                        next_stack = [n] + next_stack

        while len(next_stack) > 0:
            linker_atom, nhr_atom = next_stack.pop(0)
            pre_v.append(sorted([linker_atom, nhr_atom]))
            for frag_idx, (clique, frag) in enumerate(zip(cliques, frags)):
                if frag_idx in already:
                    continue
                if nhr_atom in clique:
                    p = mol.GetConformer().GetAtomPosition(linker_atom)
                    pos = [p.x, p.y, p.z]
                    (
                        frag_smiles,
                        frag_next,
                        frag_coords,
                        frag_neighbors,
                        frag_ori_indices,
                        frag,
                    ) = self.get_clique_mol(frag, nhr_atom, linker_atom, clique, pos)

                    if frag_idx not in already:
                        smiles.append(frag_smiles)
                        already.append(frag_idx)
                        positions.extend(frag_coords)
                        neighbors.extend(frag_neighbors)
                        ori_indices.extend(frag_ori_indices)
                        final_frags.append(frag)
                        for n in frag_next:
                            if sorted(n) not in pre_v and sorted(n) not in next_stack:
                                next_stack = [n] + next_stack
        outputs = {
            "smiles": smiles,
            "positions": positions,
            "neighbors": neighbors,
            "indices": ori_indices,
            "frags": final_frags,
            "already": already,
            "cliques": cliques,
            "atoms_ids":atoms_ids,
        }
        return outputs

    def get_root_positions(self, r1_indices, r2_indices, r3_indices, positions):
        r1_positions = np.zeros((len(r1_indices), 3))
        r2_positions = np.zeros((len(r1_indices), 3))
        r3_positions = np.zeros((len(r1_indices), 3))

        for i in range(len(r1_indices)):
            r1_positions[i] = positions[r1_indices[i]]
            r2_positions[i] = positions[r2_indices[i]]
            r3_positions[i] = positions[r3_indices[i]]

        return r1_positions, r2_positions, r3_positions

    def encode_with_unk(self, frag_smiles_list, debug=False):
        code_final = [self.atom_vocab_c2i["<SOS>"]]
        try:
            flag = False
            for frag_smiles in frag_smiles_list:
                frag_smiles = atomwise_tokenizer(frag_smiles)

                code_part = []
                for token in frag_smiles:
                    if token in self.atom_vocab_c2i.keys():
                        code_part.append(self.atom_vocab_c2i[token])
                    else:
                        flag = True
                        if "a" <= token <= "z" or "A" <= token <= "Z":
                            code_part.append(self.atom_vocab_c2i["UnkElement"])
                        else:
                            code_part.append(self.atom_vocab_c2i["UnkSymbol"])
                        if debug:
                            print("add unk here")

                code_part += [self.atom_vocab_c2i["<SEP>"]]
                code_final.extend(code_part)

            if flag and debug:
                print("final", code_final)
                print("flag", flag)

            code_final.extend([self.atom_vocab_c2i["<EOS>"], self.atom_vocab_c2i["<SEP>"]])
            code_to_return = code_final + (self.max_token_length - len(code_final)) * [self.atom_vocab_c2i["<PAD>"]]
            if sum(code_to_return[1:]) == 0:
                raise ValueError("invalid encode mol.")
            return code_to_return[: self.max_token_length], flag

        except Exception as e:
            if debug:
                print(f"there is an exception {e}")
            return None, False

    def generate_coords(self, codes, coords, r1_indices):
        new_coords = np.zeros((len(codes), 3))
        idx = 0
        for i, c in enumerate(codes):
            if c in self.ele_tokens or c == -1:
                new_coords[i] = coords[idx]
                idx += 1
            else:
                new_coords[i] = new_coords[r1_indices[i]]
                # print(c)

        return new_coords

    def rotate(self, coords, rotMat, center=(0, 0, 0)):
        """
        Rotate a selection of atoms by a given rotation around a center
        """

        newcoords = coords - center

        return np.dot(newcoords, np.transpose(rotMat)) + center

    def bond_length(self, r1_positions, positions):
        bond_lengths = np.sqrt(np.sum(np.square(np.array(positions) - np.array(r1_positions)), axis=-1))

        return bond_lengths
    
    def torch_bond_length(self, r1_positions, positions):
        bond_lengths = torch.sqrt(torch.sum(torch.square(torch.tensor(positions) - torch.tensor(r1_positions)), dim=-1))
        return bond_lengths

    def bond_angle(self, r1_positions, r2_positions, positions):
        bond_angles = np.zeros(len(positions))
        for i, cur in enumerate(positions):
            ori = r1_positions[i]
            pre = r2_positions[i]
            if np.sum(cur) == 0 or np.sum(pre) == 0 or np.sum(ori) == 0:
                continue
            cur_vec = (cur - ori) / (np.linalg.norm(cur - ori) + 1e-9) #
            pre_vec = (pre - ori) / (np.linalg.norm(pre - ori) + 1e-9) #
            cos_theta = np.arccos(np.sum(cur_vec * pre_vec))
            theta = np.degrees(cos_theta)
            bond_angles[i] = theta
        return bond_angles
    
    def torch_bond_angle(self, r1_positions, r2_positions, positions):

        cur_vec = (positions - r1_positions) / (torch.norm((positions - r1_positions).float(), dim=-1)[:, None] + 1e-9) #
        pre_vec = (r2_positions - r1_positions) / (torch.norm((r2_positions - r1_positions).float(), dim=-1)[:, None] + 1e-9) #
        cos_theta = torch.arccos(torch.sum(cur_vec * pre_vec, dim=-1))
        bond_angles = degrees(cos_theta)

        return bond_angles

    def is_inplane(self, ori, pre, cur):
        cur_vec = (cur - ori) / (np.linalg.norm(cur - ori) + 1e-9) #
        pre_vec = (pre - ori) / (np.linalg.norm(pre - ori) + 1e-9) #
        cos_theta = np.arccos(np.sum(cur_vec * pre_vec))
        theta = np.degrees(cos_theta)
        if np.abs(theta - 180) == 0:
            return False
        return True
    
    def torch_is_inplane(self, ori, pre, cur):
        cur_vec = (cur - ori) / (torch.norm((cur - ori).float(), dim=-1)[:, None] + 1e-9) #
        pre_vec = (pre - ori) / (torch.norm((pre - ori).float(), dim=-1)[:, None] + 1e-9) #
        cos_theta = torch.arccos(torch.sum(cur_vec * pre_vec, dim=-1))
        theta = degrees(cos_theta)
        return (torch.abs(theta - 180) != 0).float()


    def calculate_normal_vector(self, ori, pre, third):
        p1 = pre - ori
        p2 = third - ori
        vec1 = p1 / (np.linalg.norm(p1) + 1e-9)#
        vec2 = p2 / (np.linalg.norm(p2) + 1e-9)#
        # 叉乘: 右手法则
        normal_vector = np.cross(vec1, vec2)
        normal_vector = normal_vector / (np.linalg.norm(normal_vector) + 1e-9)#

        return normal_vector
    
    def torch_calculate_normal_vector(self, ori, pre, third):
        p1 = pre - ori
        p2 = third - ori
        vec1 = p1 / (torch.norm(p1.float(), dim=-1)[:, None] + 1e-9)#
        vec2 = p2 / (torch.norm(p2.float(), dim=-1)[:, None] + 1e-9)#
        # 叉乘: 右手法则
        normal_vector = torch.cross(vec1, vec2)
        normal_vector = normal_vector / (torch.norm(normal_vector, dim=-1)[:, None] + 1e-9)#

        return normal_vector

    def dihedral_angle(self, r1_positions, r2_positions, r3_positions, positions):
        dihedral_angles = np.zeros(len(positions))
        for i, cur in enumerate(positions):
            ori = r1_positions[i]
            pre = r2_positions[i]
            pre_pre = r3_positions[i]
            if (
                np.sum(cur) == 0.0 or np.sum(pre) == 0.0 or np.sum(ori) == 0.0 or np.sum(pre_pre) == 0.0
            ):  # 无意义，第二个，第一个 无法成角度的点,避免/0
                continue
            # exclude points that cannot be in the same plane
            isp = self.is_inplane(ori, pre, pre_pre)
            if isp is False:
                continue

            isp = self.is_inplane(ori, pre, cur)
            if isp is False:
                continue

            pre_v1 = self.calculate_normal_vector(ori, pre, pre_pre)
            pre_v2 = self.calculate_normal_vector(ori, pre, cur)
            try:
                cos_theta = np.arccos(np.sum(pre_v1 * pre_v2))
            except Exception as e:
                cos_theta = 0
            theta = np.degrees(cos_theta)
            dihedral_angles[i] = theta

        return dihedral_angles


    def torch_dihedral_angle(self, r1_positions, r2_positions, r3_positions, positions):

        pre_v1 = self.torch_calculate_normal_vector(r1_positions, r2_positions, r3_positions)
        pre_v2 = self.torch_calculate_normal_vector(r1_positions, r2_positions, positions)

        mask1 = self.torch_is_inplane(r1_positions, r2_positions, r3_positions)
        mask2 = self.torch_is_inplane(r1_positions, r2_positions, positions)

        cos_theta = torch.arccos(torch.sum(pre_v1 * pre_v2, dim=-1))
        theta = degrees(cos_theta)
        return theta * mask1 * mask2

        dihedral_angles = torch.zeros(len(positions))

        for i, cur in enumerate(positions):
            ori = r1_positions[i]
            pre = r2_positions[i]
            pre_pre = r3_positions[i]
            if (
                torch.sum(cur) == 0.0 or torch.sum(pre) == 0.0 or torch.sum(ori) == 0.0 or torch.sum(pre_pre) == 0.0
            ):  # 无意义，第二个，第一个 无法成角度的点,避免/0
                continue
            # exclude points that cannot be in the same plane
            isp = self.torch_is_inplane(ori, pre, pre_pre)
            if isp is False:
                continue

            isp = self.torch_is_inplane(ori, pre, cur)
            if isp is False:
                continue

            pre_v1 = self.torch_calculate_normal_vector(ori, pre, pre_pre)
            pre_v2 = self.torch_calculate_normal_vector(ori, pre, cur)
            try:
                cos_theta = torch.arccos(torch.sum(pre_v1 * pre_v2))
            except Exception as e:
                cos_theta = 0
            theta = torch.degrees(cos_theta)
            dihedral_angles[i] = theta

        return dihedral_angles

    def get_atom_type(self, mol):
        ssr = [list(r) for r in Chem.GetSymmSSSR(mol)]
        unique_atom_indices = set()
        for ring in ssr:
            unique_atom_indices = unique_atom_indices.union(set(ring))
        # 环系统
        ring_type = ["0"] * mol.GetNumAtoms()
        for atom_idx in unique_atom_indices:
            ring_size_list = []
            for ring in ssr:
                if atom_idx in ring:
                    ring_size_list.append(len(ring))
            ring_size_list = sorted(ring_size_list)
            ring_size_list = [str(i) for i in ring_size_list]
            if len(ring_size_list) >= 1:
                ring_type[atom_idx] = "+".join(ring_size_list)
        for atom_idx in range(mol.GetNumAtoms() - 1, -1, -1):
            atom = mol.GetAtomWithIdx(atom_idx)
            symbol = atom.GetSymbol()
            if symbol == "*":
                ring_type.pop(atom_idx)

        # 杂化系统
        hybrid_type = []
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() != "*":
                hybrid_type.append(str(atom.GetHybridization()))

        return {"ring": ring_type, "hybrid": hybrid_type}

    def atom_type(self, codes, frags):
        """检测子结构是否合理，并看元素类型是几元环"""
        frag_systems = {"ring": [], "hybrid": []}
        for frag in frags:
            atom_type = self.get_atom_type(frag)
            frag_systems["ring"].extend(deepcopy(atom_type["ring"]))
            frag_systems["hybrid"].extend(deepcopy(atom_type["hybrid"]))

        idx = 0
        mol_systems = {"ring": ["0"] * len(codes), "hybrid": [""] * len(codes)}
        for i, c in enumerate(codes):
            if c in self.ele_tokens:
                mol_systems["ring"][i] = frag_systems["ring"][idx]
                mol_systems["hybrid"][i] = frag_systems["hybrid"][idx]
                idx += 1

        return mol_systems

    def recode(self, codes, atom_types):
        fsmiles_list = []
        # ring_types = atom_types["ring"]
        # hybrid_types = atom_types["hybrid"]
        # for code, ring_type, hybrid_type in zip(codes, ring_types, hybrid_types):
        #     if code == self.atom_vocab_c2i["UnkElement"]:  # UnkElement
        #         fsmiles_list.append(self.fsmiles_vocab_c2i["UnkElement"])
        #     elif code == self.atom_vocab_c2i["UnkSymbol"]:  # UnkSymbol
        #         fsmiles_list.append(self.fsmiles_vocab_c2i["UnkSymbol"])
        #     else:
        #         if code not in self.ele_tokens:
        #            fsmiles_token = f"{self.atom_vocab_i2c[code]}"
        #         elif hybrid_type != "":
        #             fsmiles_token = f"{self.atom_vocab_i2c[code]}|{hybrid_type}_{ring_type}"
        #         else:
        #             fsmiles_token = f"{self.atom_vocab_i2c[code]}_{ring_type}"
        #         if fsmiles_token not in self.fsmiles_vocab_c2i.keys():
        #             fsmiles_list.append(self.fsmiles_vocab_c2i["UnkAtomType"])
        #         else:
        #             fsmiles_list.append(self.fsmiles_vocab_c2i[fsmiles_token])
        # fsmiles_list = fsmiles_list + [self.fsmiles_vocab_c2i["<PAD>"]] * (self.max_token_length - len(fsmiles_list))
        ring_types = atom_types["ring"]
        for code, ring_type in zip(codes, ring_types):
            if code == self.atom_vocab_c2i["UnkElement"]:  # UnkElement
                fsmiles_list.append(self.fsmiles_vocab_c2i["UnkElement"])
            elif code == self.atom_vocab_c2i["UnkSymbol"]:  # UnkSymbol
                fsmiles_list.append(self.fsmiles_vocab_c2i["UnkSymbol"])
            else:
                if code not in self.ele_tokens:
                    fsmiles_token = f"{self.atom_vocab_i2c[code]}"
                else:
                    fsmiles_token = f"{self.atom_vocab_i2c[code]}_{ring_type}"
                # if code == 6:
                #     print(self.atom_vocab_i2c[code])
                if fsmiles_token not in self.fsmiles_vocab_c2i.keys():
                    fsmiles_list.append(self.fsmiles_vocab_c2i["UnkAtomType"])
                else:
                    fsmiles_list.append(self.fsmiles_vocab_c2i[fsmiles_token])
        fsmiles_list = fsmiles_list + [self.fsmiles_vocab_c2i["<PAD>"]] * (self.max_token_length - len(fsmiles_list))

        if sum(fsmiles_list[1:]) == 0:
            raise ValueError("Invalid recode")
        return fsmiles_list

    def recode_coords(self, codes, positions, need_cross_border=False):
        # # 将越界的坐标统一置为unk_pos，并在训练数据时候忽略它
        # for i, pos in enumerate(positions):
        #     if pos[0] < 0 or pos[0] >= self.grid_size:
        #         positions[i][0] = self.unk_pos
        #     if pos[1] < 0 or pos[1] >= self.grid_size:
        #         positions[i][1] = self.unk_pos
        #     if pos[2] < 0 or pos[2] >= self.grid_size:
        #         positions[i][2] = self.unk_pos

        if need_cross_border:
            for i in range(len(positions)):
                pos = positions[i]
                if np.min(np.array(pos)) < 0 or np.max(np.array(pos)) >= self.grid_size:
                    positions[i] = [-1, -1, -1]
            return codes, positions
        # print(positions)
        left = len(codes)
        for i, pos in enumerate(positions):
            if np.min(np.array(pos)) < 0 or np.max(np.array(pos)) >= self.grid_size:
                left = i
                break
        for i in range(left, len(codes)):
            codes[i] = self.fsmiles_vocab_c2i["<PAD>"]
            positions[i][0] = 0
            positions[i][1] = 0
            positions[i][2] = 0

        return codes, positions

    def encode_mol(
        self,
        mol,
        center,
        reserve_ids=None,
        reserve_only_seed_flag=True,
        keep_bonds_prob=1.0,
        rrot=None,
        dummy_coord=None,
        need_cross_border=False,
        debug=False,
    ):
        grid_size = (self.grid_size - 1) / 2

        flatten_outputs = self.flatten_seq(
            mol,
            reserve_ids=reserve_ids,
            reserve_only_seed_flag=reserve_only_seed_flag,
            keep_bonds_prob=keep_bonds_prob,
            dummy_coord=dummy_coord,
        )
        frag_smiles_list = flatten_outputs["smiles"]
        positions = flatten_outputs["positions"]
        frags = flatten_outputs["frags"]

        # print(frag_smiles_list)
        if debug:
            print(f"frag_smiles, {frag_smiles_list}")  # 即使输入的有，这里就已经没有5，6,7,8之类的了。
        # print(flatten_outputs)
        if rrot is not None:
            positions = self.rotate(positions, rrot, center=center)
        # print(positions)
        # print(np.array(positions) - center)
        positions = (np.array(positions) - center) / self.resolution + np.array([grid_size, grid_size, grid_size])
        positions = np.rint(positions).astype("int")
        # print(positions)
        codes, _ = self.encode_with_unk(frag_smiles_list)  # 这里有元素、符号层面的过滤
    
        if debug:
            print(f"codes, {codes}")
            print(f"code_tokens, {[self.atom_vocab_i2c[c] for c in codes]}")
        atom_types = self.atom_type(codes, frags)  # 这里有旋转键、螺环桥环、大环小环之类的过滤
        if debug:
            for k, v in atom_types.items():
                print(f"atom_types: {k}, {v}")
        # print(atom_types)
        fsmiles = self.recode(codes, atom_types)
        # print(fsmiles)
        if debug:
            print(f"fsmiles, {fsmiles}")
            print(f"fsmiles_tokens, {[self.fsmiles_vocab_i2c[f] for f in fsmiles]}")
        # print(codes)
        # print(fsmiles)
        # codes=fsmiles
        # codes = [1, 4, 4, 77, 254, 3, 123, 244, 144, 123, 255, 116, 245, 153, 113, 255, 135, 116, 245, 123, 244, 254, 3, 4, 3, 52, 4, 250, 243, 77, 251, 77, 4, 4, 3, 4, 254, 3, 123, 244, 123, 144, 123, 255, 144, 123, 244, 3, 91, 4, 3, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # codes = [1, 4, 4, 6, 4, 35, 28, 6, 36, 5, 39, 3, 13, 29, 14, 13, 30, 13, 40, 13, 40, 14, 13, 40, 13, 30, 16, 29, 3, 4, 39, 3, 13, 29, 13, 14, 13, 40, 14, 13, 29, 3, 7, 4, 3, 6, 4, 4, 3, 4, 3, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # print(fsmiles)
        r1_indices, r2_indices, r3_indices = find_root_smi(codes, self)
        # print(r1_indices)
        # print(r2_indices)
        # print(positions)
        # print(r3_indices)
        if debug:
            print(f"r1_indices, {r1_indices}")
            print(f"r2_indices, {r2_indices}")
            print(f"r3_indices, {r3_indices}")
        # print(r1_indices)
        # assert 0
        positions = self.generate_coords(codes, positions, r1_indices)
        
        r1_positions, r2_positions, r3_positions = self.get_root_positions(
            r1_indices, r2_indices, r3_indices, positions
        )
        
        bond_lengths = self.bond_length(r1_positions, positions)
        bond_lengths = np.clip(np.rint(bond_lengths).astype("int"), a_min=0, a_max=21)
        bond_lengths = np.array([self.bond_length_grid[d] for d in bond_lengths])
        bond_angles = self.bond_angle(r1_positions, r2_positions, positions)
        bond_angles = np.clip(np.rint(bond_angles).astype("int"), a_max=180, a_min=0)
        dihedral_angles = self.dihedral_angle(r1_positions, r2_positions, r3_positions, positions)
        dihedral_angles = np.clip(np.rint(dihedral_angles).astype("int"), a_min=0, a_max=180)

        # print(bond_lengths)
        # 移除越界原子

        fsmiles, positions = self.recode_coords(fsmiles, positions, need_cross_border)

        code_outputs = {
            "ligand_tokens": np.array(fsmiles),
            "ligand_coords": np.array(positions),
            "ligand_r1_indices": np.array(r1_indices),
            "ligand_r2_indices": np.array(r2_indices),
            "ligand_r3_indices": np.array(r3_indices),
            "bond_angles": np.array(bond_angles),
            "bond_lengths": np.array(bond_lengths),
            "dihedral_angles": np.array(dihedral_angles),
            "cliques": flatten_outputs['cliques'],
            "atoms_ids": flatten_outputs['atoms_ids'],
            "frags": frags,
            # "check_frags": flatten_outputs['check_frag']

        }

        # for frag in frags:
        #     print(Chem.MolToSmiles(frag))
        # print('********')
        return code_outputs

    def encode_mol_2d(self, mol, debug=False):
        flatten_outputs = self.flatten_seq(mol)
        frag_smiles_list, frags = flatten_outputs["smiles"], flatten_outputs["frags"]
        if debug:
            print(f"frag_smiles, {frag_smiles_list}")
        codes, _ = self.encode_with_unk(frag_smiles_list)
        if debug:
            print(f"codes, {codes}")
            print(f"code_tokens, {[self.atom_vocab_i2c[c] for c in codes]}")
        atom_types = self.atom_type(codes, frags)
        if debug:
            for k, v in atom_types.items():
                print(f"atom_types: {k}, {v}")
        fsmiles = self.recode(codes, atom_types)
        if debug:
            print(f"fsmiles, {fsmiles}")
            print(f"fsmiles_tokens, {[self.fsmiles_vocab_i2c[f] for f in fsmiles]}")

        return fsmiles

    def merge_smiles(self, rwmol, smi, uuid, debug=False, post_process=True):
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            # label conn atom
            nbr_idx = -1
            orig_idx = -1

            for i in range(rwmol.GetNumAtoms() - 1, -1, -1):
                atom = rwmol.GetAtomWithIdx(i)
                s = atom.GetSymbol()
                if s == "*":
                    nbr_idx = atom.GetIdx()
                    n = atom.GetNeighbors()
                    nbr = n[0]
                    orig_idx = nbr.GetIdx()
                    atom1 = rwmol.GetAtomWithIdx(orig_idx)
                    atom1.SetProp("delete", str(uuid))
                    break

            if nbr_idx == -1:
                return rwmol, True  # return modified mol, isFinish true
            # print(mol)
            # remove fake atom
            rwmol.RemoveAtom(nbr_idx)
            pre = rwmol.GetNumAtoms()
            if mol is None:
                return None, True
            combo = Chem.CombineMols(rwmol, mol)

            # connect two atoms
            # find previous original atoms
            pre_index = -1
            for i in range(rwmol.GetNumAtoms()):
                atom = rwmol.GetAtomWithIdx(i)
                labels = atom.GetPropsAsDict()
                if "delete" not in labels.keys():
                    continue
                label = labels["delete"]
                if str(label) == str(uuid):
                    pre_index = atom.GetIdx()
                    break

            # connect
            ecombo = Chem.RWMol(combo)
            ecombo.AddBond(pre_index, pre, Chem.BondType.SINGLE)
            ecombo.UpdatePropertyCache(strict=False)
            # print(ecombo)
            if post_process:
                for linker_id in [pre_index, pre]:
                    linker_atom = ecombo.GetAtomWithIdx(linker_id)
                    norm_degree = linker_atom.GetDegree() + linker_atom.GetNumExplicitHs()
                    if linker_atom.GetAtomicNum() == 7 and norm_degree == 5:
                        origin_degree = 4
                        Hs_exceed = linker_atom.GetDegree() + linker_atom.GetNumExplicitHs() - origin_degree
                        if Hs_exceed > 0:
                            linker_atom.SetNumExplicitHs(linker_atom.GetNumExplicitHs() - Hs_exceed)
                    elif linker_atom.GetAtomicNum() == 6 and norm_degree == 5:
                        origin_degree = 4
                        Hs_exceed = linker_atom.GetDegree() + linker_atom.GetNumExplicitHs() - origin_degree
                        if Hs_exceed > 0:
                            linker_atom.SetNumExplicitHs(linker_atom.GetNumExplicitHs() - Hs_exceed)
                    elif (linker_atom.GetAtomicNum() in (6, 7, 8, 16)) and linker_atom.HasValenceViolation():
                        h_nums = linker_atom.GetNumExplicitHs()
                        for cur_num in range(h_nums - 1, -1, -1):
                            if linker_atom.HasValenceViolation():
                                linker_atom.SetNumExplicitHs(cur_num)
            # print('ok')
            return ecombo, False
        except Exception as e:
            if debug:
                print(f"[mergeSmi] invalid {e} {traceback.format_exc()}")
            return None, True

    def merge_smiles_3d(self, res, position, debug=False, use_star_checker=True):
        try:
            mol = Chem.MolFromSmiles(res[0])
            m = Chem.RWMol(mol)
            # print(mol)
            for i, r in enumerate(res):
                if i == 0:
                    continue
                m, isFinish = self.merge_smiles(m, r, i, debug)
                # print(m)
                if m is None:
                    if debug:
                        print("[merge_smiles_3d] mol is None")
                    return None, None, "merge_smiles_error"
                if isFinish:
                    break

            # delete unnecessary fake atom
            if use_star_checker:
                fakes = []
                for i in range(m.GetNumAtoms()):
                    atom = m.GetAtomWithIdx(i)
                    s = atom.GetSymbol()
                    if s == "*":
                        fakes.append(atom.GetIdx())
                if len(fakes) > 0:
                    return None, None, "star_checker_error"

                fakes = sorted(fakes, reverse=True)
                for f in fakes:
                    # m.ReplaceAtom(f, Chem.Atom('H'))
                    m.RemoveAtom(f)
                Chem.SanitizeMol(m)
                m = Chem.RemoveHs(m)

                if m.GetNumAtoms() != len(position):
                    if debug:
                        print(
                            f"[merge_smiles_3d] atoms != position, position: {len(position)}, atoms: {m.GetNumAtoms()}"
                        )
                    return None, None, "star_checker_error"

            conf = Chem.Conformer(m.GetNumAtoms())
            pos_idx = 0
            for i in range(m.GetNumAtoms()):
                if not use_star_checker:
                    atom_symbol = m.GetAtomWithIdx(i).GetSymbol()
                    if atom_symbol == "*":
                        continue
                pos = position[pos_idx]
                p = rdGeometry.Point3D(pos[0], pos[1], pos[2])
                conf.SetAtomPosition(i, p)
                pos_idx += 1
            m.AddConformer(conf)
            m.SetProp("_Name", Chem.MolToSmiles(m))
            Chem.SanitizeMol(m)
            if m is None:
                return None, None, "rdkit_error"
            return Chem.MolToSmiles(m), m, "merge_smiles_sucess"
        except Exception as e:
            if debug:
                print(f"invalid {e} {traceback.format_exc()}")
            return None, None, "rdkit_error"

    def decode_3d(
        self,
        batch_codes,
        batch_positions,
        batch_code_probs=None,
        debug=False,
        use_star_checker=True,
        return_error_reason=False,
    ):
        decoded_tokens, decoded_smiles, decoded_positions = [], [], []
        if batch_code_probs is not None:
            decode_code_probs = []
        batch_positions = batch_positions.astype(np.float16).tolist()
        for i, sample in enumerate(batch_codes):
            # sample是分子FSMILES序列的索引序列
            try:
                cur_smiles, cur_positions, cur_tokens, cur_frag_smiles = "", [], [], []
                sample_postions = batch_positions[i]
                if batch_code_probs is not None:
                    cur_token_probs = []
                    sample_code_probs = batch_code_probs[i]
                for j, fsmiles_idx in enumerate(sample[:]):
                    # fsmiles_idx是FSMILES token的索引
                    fsmiles_char = self.fsmiles_vocab_i2c[fsmiles_idx]
                    atom_symbol = fsmiles_char.split("_")[0].split("|")[0]
                    if fsmiles_idx == self.fsmiles_vocab_c2i["<EOS>"]:
                        cur_tokens.append(fsmiles_char)
                        if batch_code_probs is not None:
                            cur_token_probs.append(sample_code_probs[j])
                        break
                    if fsmiles_idx in (
                        self.fsmiles_vocab_c2i["<SOS>"],
                        self.fsmiles_vocab_c2i["<S2OS>"],
                        self.fsmiles_vocab_c2i["<S3OS>"],
                    ):
                        cur_tokens.append(fsmiles_char)
                        if batch_code_probs is not None:
                            cur_token_probs.append(sample_code_probs[j])
                        continue
                    cur_tokens.append(fsmiles_char)
                    if batch_code_probs is not None:
                        cur_token_probs.append(sample_code_probs[j])
                    if fsmiles_idx == self.fsmiles_vocab_c2i["<SEP>"]:
                        cur_frag_smiles.append(deepcopy(cur_smiles))
                        cur_smiles = ""
                        continue
                    cur_smiles += atom_symbol
                    if self.atom_vocab_c2i[atom_symbol] in self.ele_tokens:
                        cur_positions.append(sample_postions[j])
                decoded_tokens.append(deepcopy(cur_tokens))
                decoded_smiles.append(deepcopy(cur_frag_smiles))
                decoded_positions.append(deepcopy(cur_positions))
                if batch_code_probs is not None:
                    decode_code_probs.append(deepcopy(cur_token_probs))
            except Exception as e:
                if debug:
                    print(f"decode {e}")
                continue
        # 解码分子
        gen_smiles, gen_mols = [], []
        merge_smiles_error, star_checker_error, rdkit_error = 0, 0, 0
        print(decoded_smiles, '****')
        for i, cur_frag_smiles in enumerate(decoded_smiles):
            try:
                if len(cur_frag_smiles) > 0:
                    smi, mol, error_reason = self.merge_smiles_3d(
                        cur_frag_smiles,
                        decoded_positions[i],
                        debug=debug,
                        use_star_checker=use_star_checker,
                    )
                    print(smi, mol, error_reason, '****')
                    if "merge_smiles_error" == error_reason:
                        merge_smiles_error += 1
                    elif "star_checker_error" == error_reason:
                        star_checker_error += 1
                    elif "rdkit_error" == error_reason:
                        rdkit_error += 1
                    gen_smiles.append(deepcopy(smi))
                    gen_mols.append(deepcopy(mol))
                else:
                    gen_smiles.append(None)
                    gen_mols.append(None)
            except Exception as e:
                gen_smiles.append(None)
                gen_mols.append(None)
                if debug:
                    print(f"merge smiles {e}")

        outputs = {"smiles": gen_smiles, "tokens": decoded_tokens, "mols": gen_mols}
        if batch_code_probs is not None:
            outputs["token_probs"] = decode_code_probs
        if return_error_reason:
            smiles_gen_remain = len([item for item in decoded_smiles if len(item) > 0])
            merge_smiles_remain = smiles_gen_remain - merge_smiles_error
            star_checker_remain = merge_smiles_remain - star_checker_error
            rdkit_remain = star_checker_remain - rdkit_error
            error_reason = {
                "smiles_gen_remain": smiles_gen_remain,
                "merge_smiles_remain": merge_smiles_remain,
                "star_checker_remain": star_checker_remain,
                "rdkit_remain": rdkit_remain,
            }
            outputs["error_reasons"] = error_reason

        return outputs
