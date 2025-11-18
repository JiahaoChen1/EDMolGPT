#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
from functools import lru_cache
# from .ligand_code_util import LigandCodeUtil
import numpy as np


@lru_cache()
def get_atom_non_element_tokens(cls_object):
    # from ligand_code_util import LigandCodeUtil

    ele_tokens = set(cls_object.ele_tokens)
    bracket_pre = {cls_object.atom_vocab_c2i["("]}
    bracket_post = {cls_object.atom_vocab_c2i[")"]}
    sep = {cls_object.atom_vocab_c2i["<SEP>"]}

    return ele_tokens, bracket_pre, bracket_post, sep


def find_in_other_frag(sample, step_idx, star_single, cls_object):
    # from ligand_code_util import LigandCodeUtil
    (
        ele_tokens,
        bracket_pre,
        bracket_post,
        _,
    ) = get_atom_non_element_tokens(cls_object)

    pre_idx = 0
    try:  # 以防idx是空的
        star_idx = np.where(star_single[:step_idx] == 1)[-1][-1]
        step_idx = star_idx - 1
        while step_idx >= 0:
            # 不可能再跨了
            if sample[step_idx] in ele_tokens:
                pre_idx = step_idx
                break
            if sample[step_idx] in bracket_post:
                branch = []
                branch.append(deepcopy(step_idx))
                k = step_idx - 1
                flag = False
                while k >= 0 and len(branch) > 0:
                    if sample[k] in bracket_post:
                        branch.append(deepcopy(k))
                        k -= 1
                        continue
                    if sample[k] in bracket_pre:
                        code_idx = branch.pop(-1)
                        if sample[k] + 1 != sample[code_idx]:
                            print("grammer error!", sample[k], sample[code_idx], branch)
                            break
                        if sample[k] + 1 == sample[code_idx] and len(branch) == 0:
                            flag = True
                            break
                    k -= 1
                if flag and k - 1 >= 0:
                    step_idx = k - 1
                    continue
                else:
                    break
            step_idx = step_idx - 1
    except Exception as e:
        return pre_idx, 0

    return pre_idx, star_idx


def find_root_smi_cur(sample, step_idx, star_mask, cls_object):
    """
    因为是给非元素token找的
    :param sample:
    :param step_idx:  单条step_idx是应该有顺序的
    :param star:
    :return:

    """
    ele_tokens, bracket_pre, bracket_post, sep = get_atom_non_element_tokens(cls_object)

    pre_idx = 0
    if step_idx == 0:
        return pre_idx, star_mask
    if sample[step_idx] not in ele_tokens:
        j = step_idx
    else:
        j = step_idx - 1

    while j >= 0:
        if sample[j] in sep and j != 0:
            """
            跨范围的一般不是特殊符号
            """
            pre_idx, star_idx = find_in_other_frag(sample, j, star_mask, cls_object)
            star_mask[star_idx] += 1
            break
        if sample[j] in ele_tokens:
            pre_idx = j
            break
        if sample[j] in bracket_post:
            branch = []
            branch.append(deepcopy(j))
            k = j - 1
            flag = False
            while k >= 0 and len(branch) > 0:
                if sample[k] in bracket_post:
                    branch.append(deepcopy(k))
                    k -= 1
                    continue
                if sample[k] in bracket_pre:
                    code_idx = branch.pop(-1)
                    if sample[k] + 1 != sample[code_idx]:
                        print("grammer err !", k, code_idx, sample[k], sample[code_idx], branch)
                        break
                    if sample[k] + 1 == sample[code_idx] and len(branch) == 0:
                        flag = True
                        break
                k -= 1
            if flag and k - 1 >= 0:
                j = k - 1
                continue
            else:
                break
        j = j - 1

    return pre_idx, star_mask


def find_root_smi(sample, cls_object):
    
    ele_tokens = set(cls_object.ele_tokens)
    star_mask = np.where(
        (np.array(sample) == cls_object.atom_vocab_c2i["[*]"])
        | (np.array(sample) == cls_object.atom_vocab_c2i["([*])"]),
        1,
        0,
    )
    r1_indices = np.zeros(len(sample), dtype=int)
    r2_indices = np.zeros((len(sample)), dtype=int)
    r3_indices = np.zeros((len(sample)), dtype=int)
    for step_idx, code in enumerate(sample):
        r1_idx, star_mask_tmp = find_root_smi_cur(sample, step_idx, deepcopy(star_mask), cls_object)
        r1_indices[step_idx] = r1_idx
        if code in ele_tokens:
            r2_idx = r1_indices[r1_idx]
            r2_indices[step_idx] = r2_idx
            r3_idx = r1_indices[r2_idx]
            r3_indices[step_idx] = r3_idx
            star_mask = star_mask_tmp
        else:
            r2_indices[step_idx] = r1_idx
            r3_indices[step_idx] = r1_idx

    return r1_indices.tolist(), r2_indices.tolist(), r3_indices.tolist()