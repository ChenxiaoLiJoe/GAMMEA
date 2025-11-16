
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math

from transformers.activations import ACT2FN
from transformers.pytorch_utils import apply_chunking_to_forward

from .layers import ProjectionHead
from .Tool_model import GAT, GCN
import pdb

from .submodels import AuxiliaryNet, BackboneNet, MLP


# 门控0和1的图像模态
class MformerFusion_Gated_w_Img(nn.Module):
    def __init__(self, args, modal_num, with_weight=1):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        self.fusion_layer = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])
        self.image_gate = AuxiliaryNet(args.auxiliary_hidden_size, args.embedding_length)

        # self.type_embedding = nn.Embedding(args.inner_view_num, args.hidden_size)
        self.type_id = torch.tensor([0, 1, 2, 3, 4, 5]).cuda()

    def forward(self, embs):
        # 清洗有效模态
        embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]
        modal_num = len(embs)
        bs = embs[0].size(0)

        # 计算门控，保证维度一致 [B]
        gates = []
        for i in range(modal_num):
            if i == 0:
                g = self.image_gate(embs[0])   # 可能输出 [B,1] 或 [1]
                if g.dim() == 2 and g.size(1) == 1:
                    g = g.squeeze(1)           # [B]
                if g.dim() == 0:               # 标量 → 扩展到 batch
                    g = g.expand(bs)
                elif g.size(0) == 1:           # 单元素 → 扩展到 batch
                    g = g.expand(bs)
            else:
                g = torch.ones(bs, device=embs[i].device)
            gates.append(g)

        # 堆叠模态嵌入 → [B, modal_num, seq_len, hidden_size]
        hidden_states = torch.stack(embs, dim=1)

        # Transformer 融合
        for i, layer_module in enumerate(self.fusion_layer):
            layer_outputs = layer_module(hidden_states, output_attentions=True)
            hidden_states = layer_outputs[0]
        # torch.Size([30355, 5, 4, 4])
        # attention_pro = layer_outputs[1]
        # torch.Size([30355, 4, 4])

        # 提取注意力权重 → [B, modal_num]
        attention_pro = torch.sum(layer_outputs[1], dim=-3)
        attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(modal_num * self.args.num_attention_heads)
        weight_norm = F.softmax(attention_pro_comb, dim=-1)

        # 堆叠成 [B, modal_num]
        gate_vec = torch.stack(gates, dim=1)

        # 先与门控相乘，使被关掉的模态不参与加权?
        masked_weight = weight_norm * gate_vec  # [B, modal_num]

        # 可选：对存活模态重归一化，避免总权重缩小（推荐）?
        denom = masked_weight.sum(dim=1, keepdim=True).clamp(min=1e-8)
        masked_weight = masked_weight / denom  # 只在有效模态上归一化

        # 使用注意力权重对模态嵌入加权
        embs = [masked_weight[:, idx].unsqueeze(1) * F.normalize(embs[idx]) for idx in range(modal_num)]

        # 拼接成联合表示 → [B, modal_num * hidden_size]
        joint_emb = torch.cat(embs, dim=1)

        return joint_emb, hidden_states, weight_norm


# 软门控0和1的模态
class MformerFusion_Soft_Gated_w(nn.Module):
    def __init__(self, args, modal_num, with_weight=1):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        self.fusion_layer = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])
        self.modality_gates = nn.ModuleList([AuxiliaryNet(args.auxiliary_hidden_size, args.embedding_length) for _ in range(modal_num)])

        # self.type_embedding = nn.Embedding(args.inner_view_num, args.hidden_size)
        self.type_id = torch.tensor([0, 1, 2, 3, 4, 5]).cuda()

    def forward(self, embs):
        # 清洗有效模态
        embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]
        modal_num = len(embs)

        # 计算门控 g_m ∈ {0,1} 或 [0,1]（按需选择 hard/soft）？维度？
        gates = [
            self.modality_gates[i](embs[i])
            for i in range(modal_num)
        ]

        # 堆叠模态嵌入 → [B, modal_num, seq_len, hidden_size]
        hidden_states = torch.stack(embs, dim=1)
        bs = hidden_states.shape[0]

        # Transformer 融合
        for i, layer_module in enumerate(self.fusion_layer):
            layer_outputs = layer_module(hidden_states, output_attentions=True)
            hidden_states = layer_outputs[0]
        # torch.Size([30355, 5, 4, 4])
        # attention_pro = layer_outputs[1]
        # torch.Size([30355, 4, 4])

        # 提取注意力权重 → [B, modal_num]
        attention_pro = torch.sum(layer_outputs[1], dim=-3)
        attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(modal_num * self.args.num_attention_heads)
        weight_norm = F.softmax(attention_pro_comb, dim=-1)

        # 将 [B,1,1] 的 gate 挤压到 [B,1] 以匹配注意力权重，再扩回?
        # gate_vec = torch.stack([g.squeeze(-1).squeeze(-1) for g in gates], dim=1)  # [B, modal_num]

        processed_gates = []
        for g in gates:
            # 如果是标量，扩展成 [B]，假设 batch_size=1 时也能兼容
            if g.dim() == 0:
                g = g.unsqueeze(0)
            # 如果是二维 [B,1]，压缩成 [B]
            if g.dim() == 2 and g.size(1) == 1:
                g = g.squeeze(1)
            processed_gates.append(g)

        # 堆叠成 [B, modal_num]
        gate_vec = torch.stack(processed_gates, dim=1)

        # 先与门控相乘，使被关掉的模态不参与加权?
        masked_weight = weight_norm * gate_vec  # [B, modal_num]

        # 可选：对存活模态重归一化，避免总权重缩小（推荐）?
        denom = masked_weight.sum(dim=1, keepdim=True).clamp(min=1e-8)
        masked_weight = masked_weight / denom  # 只在有效模态上归一化

        # 使用注意力权重对模态嵌入加权
        embs = [masked_weight[:, idx].unsqueeze(1) * F.normalize(embs[idx]) for idx in range(modal_num)]

        # 拼接成联合表示 → [B, modal_num * hidden_size]
        joint_emb = torch.cat(embs, dim=1)

        return joint_emb, hidden_states, weight_norm


# 门控0和1的模态
class MformerFusion_Gated_w(nn.Module):
    def __init__(self, args, modal_num, with_weight=1):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        self.fusion_layer = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])
        self.modality_gates = nn.ModuleList([AuxiliaryNet(args.auxiliary_hidden_size, args.embedding_length) for _ in range(modal_num)])

        # self.type_embedding = nn.Embedding(args.inner_view_num, args.hidden_size)
        self.type_id = torch.tensor([0, 1, 2, 3, 4, 5]).cuda()

    def forward(self, embs):
        # 清洗有效模态
        embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]
        modal_num = len(embs)

        # 计算门控 g_m ∈ {0,1} 或 [0,1]（按需选择 hard/soft）？维度？
        gates = [
            (self.modality_gates[i](embs[i]) > 0.5).float()
            for i in range(modal_num)
        ]

        # 堆叠模态嵌入 → [B, modal_num, seq_len, hidden_size]
        hidden_states = torch.stack(embs, dim=1)
        bs = hidden_states.shape[0]

        # Transformer 融合
        for i, layer_module in enumerate(self.fusion_layer):
            layer_outputs = layer_module(hidden_states, output_attentions=True)
            hidden_states = layer_outputs[0]
        # torch.Size([30355, 5, 4, 4])
        # attention_pro = layer_outputs[1]
        # torch.Size([30355, 4, 4])

        # 提取注意力权重 → [B, modal_num]
        attention_pro = torch.sum(layer_outputs[1], dim=-3)
        attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(modal_num * self.args.num_attention_heads)
        weight_norm = F.softmax(attention_pro_comb, dim=-1)

        # 将 [B,1,1] 的 gate 挤压到 [B,1] 以匹配注意力权重，再扩回?
        # gate_vec = torch.stack([g.squeeze(-1).squeeze(-1) for g in gates], dim=1)  # [B, modal_num]

        processed_gates = []
        for g in gates:
            # 如果是标量，扩展成 [B]，假设 batch_size=1 时也能兼容
            if g.dim() == 0:
                g = g.unsqueeze(0)
            # 如果是二维 [B,1]，压缩成 [B]
            if g.dim() == 2 and g.size(1) == 1:
                g = g.squeeze(1)
            processed_gates.append(g)

        # 堆叠成 [B, modal_num]
        gate_vec = torch.stack(processed_gates, dim=1)

        # 先与门控相乘，使被关掉的模态不参与加权?
        masked_weight = weight_norm * gate_vec  # [B, modal_num]

        # 可选：对存活模态重归一化，避免总权重缩小（推荐）?
        denom = masked_weight.sum(dim=1, keepdim=True).clamp(min=1e-8)
        masked_weight = masked_weight / denom  # 只在有效模态上归一化

        # 使用注意力权重对模态嵌入加权
        embs = [masked_weight[:, idx].unsqueeze(1) * F.normalize(embs[idx]) for idx in range(modal_num)]

        # 拼接成联合表示 → [B, modal_num * hidden_size]
        joint_emb = torch.cat(embs, dim=1)

        return joint_emb, hidden_states, weight_norm


# 门控1的模态
class MformerFusion_Gated_wo(nn.Module):
    def __init__(self, args, modal_num, with_weight=1):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        self.fusion_layer = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])
        self.modality_gates = nn.ModuleList([AuxiliaryNet(args.auxiliary_hidden_size, args.embedding_length) for _ in range(modal_num)])

        # self.type_embedding = nn.Embedding(args.inner_view_num, args.hidden_size)
        self.type_id = torch.tensor([0, 1, 2, 3, 4, 5]).cuda()

    def forward(self, embs):
        # 清洗有效模态
        embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]
        modal_num = len(embs)

        # 计算门控 g_m ∈ {0,1} 或 [0,1]（按需选择 hard/soft）？维度？
        gates = [
            (self.modality_gates[i](embs[i](dim=1, keepdim=True)) > 0.5).float()  # hard gate: 二值
            # self.modality_gates[i](embs[i].mean(dim=1, keepdim=True))               # soft gate: 连续
            for i in range(modal_num)
        ]

        # 将 [B,1,1] 的 gate 挤压到 [B,1] 以匹配注意力权重，再扩回?
        # gate_vec = torch.stack([g.squeeze(-1).squeeze(-1) for g in gates], dim=1)  # [B, modal_num]

        processed_gates = []
        for g in gates:
            # 如果是标量，扩展成 [B]，假设 batch_size=1 时也能兼容
            if g.dim() == 0:
                g = g.unsqueeze(0)
            # 如果是二维 [B,1]，压缩成 [B]
            if g.dim() == 2 and g.size(1) == 1:
                g = g.squeeze(1)
            processed_gates.append(g)

        # 堆叠成 [B, modal_num]
        gate_vec = torch.stack(processed_gates, dim=1)

        # 如果门控全为0，强制保留modality_gates得到的最大概率的模态
        if sum([gates[i].sum() for i in range(modal_num)]) == 0:
            # 计算每个模态的门控概率值（在阈值化之前）
            gate_probs = []
            for i in range(modal_num):
                # 获取原始概率值（未经过阈值处理）
                prob = self.modality_gates[i](embs[i].mean(dim=1, keepdim=True))
                gate_probs.append(prob.squeeze(-1).squeeze(-1))  # 调整为 [B]

            # 堆叠所有模态的概率值
            gate_probs_stacked = torch.stack(gate_probs, dim=1)  # [B, modal_num]

            # 找到每个样本中概率最大的模态索引
            max_prob_indices = gate_probs_stacked.argmax(dim=1)  # [B]

            # 为每个样本强制开启概率最大的模态
            for batch_idx in range(gate_vec.shape[0]):
                gate_vec[batch_idx, max_prob_indices[batch_idx]] = 1.0

        # 模态嵌入与门控相乘，将门控为0的模态置零，目的是控制计算效率，避免冗余计算
        embs = [embs[i] * gates[i] for i in range(modal_num)]

        # 二次清洗，将嵌入为0的嵌入清洗
        embs = [embs[i] for i in range(modal_num) if embs[i].sum() != 0]

        # 堆叠模态嵌入 → [B, modal_num, seq_len, hidden_size]
        hidden_states = torch.stack(embs, dim=1)
        bs = hidden_states.shape[0]

        # Transformer 融合
        for i, layer_module in enumerate(self.fusion_layer):
            layer_outputs = layer_module(hidden_states, output_attentions=True)
            hidden_states = layer_outputs[0]
        # torch.Size([30355, 5, 4, 4])
        # attention_pro = layer_outputs[1]
        # torch.Size([30355, 4, 4])

        # 提取注意力权重 → [B, modal_num]
        attention_pro = torch.sum(layer_outputs[1], dim=-3)
        attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(modal_num * self.args.num_attention_heads)
        weight_norm = F.softmax(attention_pro_comb, dim=-1)

        # 先与门控相乘，使被关掉的模态不参与加权?
        masked_weight = weight_norm * gate_vec  # [B, modal_num]

        # 可选：对存活模态重归一化，避免总权重缩小（推荐）?
        denom = masked_weight.sum(dim=1, keepdim=True).clamp(min=1e-8)
        masked_weight = masked_weight / denom  # 只在有效模态上归一化

        # 使用注意力权重对模态嵌入加权
        embs = [masked_weight[:, idx].unsqueeze(1) * F.normalize(embs[idx]) for idx in range(modal_num)]

        # 拼接成联合表示 → [B, modal_num * hidden_size]
        joint_emb = torch.cat(embs, dim=1)

        return joint_emb, hidden_states, weight_norm


class MformerFusion_doublelinear_hiddenstates_jzbert(nn.Module):
    def __init__(self, args, modal_num, with_weight=1):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        self.fusion_layer = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])
        self.ConfidNet = nn.Sequential(
            nn.Linear(300, 300 * 2),
            nn.Linear(300 * 2, 300),
            nn.Linear(300, 1),
            nn.Sigmoid()
        )

    def forward(self, embs):
        device = embs[0].device  # 获取第一个张量的设备
        embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]
        modal_num = len(embs)

        hidden_states = torch.stack(embs, dim=1).to(device)
        bs = hidden_states.shape[0]

        tcps = []
        for idx in range(modal_num):
            tcp = self.ConfidNet(hidden_states[:, idx, :]).squeeze()
            tcps.append(tcp.detach())

        holos = []
        for i, tcp in enumerate(tcps):
            current_tcp = tcps[0]  # 第一个元素
            # ap_tcp = torch.prod(torch.tensor(tcps))  # 所有元素的乘积
            ap_tcp = torch.prod(torch.stack(tcps)).to(device)  # 所有元素的乘积
            holo = torch.log(current_tcp) / (torch.log(ap_tcp) + 1e-8)
            holos.append(holo.detach())

        # 将 tcps 和 holos 中的元素逐位相加，并分别使用 detach()
        cb = []
        for tcp, holo in zip(tcps, holos):
            cb.append((tcp.detach() + holo.detach()).tolist())

        # 使用 torch.stack 将 cb 中的所有元素按照维度为1进行堆叠
        # weight_norm_linear = torch.stack([torch.tensor(item).to(device) for item in cb], dim=1)
        # softmax = nn.Softmax(1)
        # weight_norm_linear = softmax(weight_norm_linear)

        for i, layer_module in enumerate(self.fusion_layer):
            layer_outputs = layer_module(hidden_states, output_attentions=True)
            hidden_states = layer_outputs[0]
        attention_pro = torch.sum(layer_outputs[1], dim=-3)
        attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(modal_num * self.args.num_attention_heads)
        # weight_norm_bert = F.softmax(attention_pro_comb, dim=-1)

        # dus = torch.mean(torch.abs(weight_norm_bert - 1 / weight_norm_bert.shape[1]), dim=1, keepdim=True).detach()
        dus = torch.mean(torch.abs(attention_pro_comb - 1 / attention_pro_comb.shape[1]), dim=1, keepdim=True).detach()
        # dus = torch.mean(torch.abs(weight_norm_bert - 1 / weight_norm_bert.shape[1]), dim=1, keepdim=False)
        # dus = dus.unsqueeze(1).expand(-1, modal_num)
        # assert modal_num == dus.shape[1], f"modal_num ({modal_num}) does not match dus shape ({dus.shape})"

        current_modal_du_power = torch.pow(dus, modal_num).detach()
        other_modal_product = torch.prod(dus, dim=1).detach()
        # other_modal_product = torch.prod(dus, dim=1, keepdim=True).expand(-1, modal_num)
        condition = current_modal_du_power > other_modal_product

        # 初始化 rcs 列表
        rcs = []

        # 计算每个模态的 rc
        for i in range(modal_num):
            # current_du = dus[:, i]
            current_du = dus.squeeze(1)
            other_dus = torch.cat([dus[:, :i], dus[:, (i + 1):]], dim=1)
            other_product = torch.prod(other_dus, dim=1, keepdim=True)

            rc = torch.where(condition[:, i].unsqueeze(1),
                             torch.ones_like(current_du).unsqueeze(1),
                             current_du.unsqueeze(1) / other_product)
            rcs.append(rc)

        # 将所有模态的 rc 合并
        rcs = torch.cat(rcs, dim=1)

        ccbs = []
        for i in range(modal_num):
            cb_tensor = torch.tensor(cb[i]).to(device)
            rc_tensor = rcs[:, i]
            ccb = cb_tensor * rc_tensor
            ccbs.append(ccb)

        # ccbs = torch.stack(ccbs, dim=1)
        # print(ccbs)
        weight_norm = torch.stack(ccbs, 1)
        softmax = nn.Softmax(1)
        weight_norm = softmax(weight_norm)

        embs = [weight_norm[:, idx].unsqueeze(1) * F.normalize(embs[idx]) for idx in range(modal_num)]
        joint_emb = torch.cat(embs, dim=1)

        return joint_emb, hidden_states, weight_norm


class MformerFusion_Original(nn.Module):
    def __init__(self, args, modal_num, with_weight=1):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        self.fusion_layer = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])
        # self.type_embedding = nn.Embedding(args.inner_view_num, args.hidden_size)
        self.type_id = torch.tensor([0, 1, 2, 3, 4, 5]).cuda()

    def forward(self, embs):
        embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]
        modal_num = len(embs)

        hidden_states = torch.stack(embs, dim=1)
        bs = hidden_states.shape[0]
        for i, layer_module in enumerate(self.fusion_layer):
            layer_outputs = layer_module(hidden_states, output_attentions=True)
            hidden_states = layer_outputs[0]
        # torch.Size([30355, 5, 4, 4])
        # attention_pro = layer_outputs[1]
        # torch.Size([30355, 4, 4])
        attention_pro = torch.sum(layer_outputs[1], dim=-3)
        attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(modal_num * self.args.num_attention_heads)
        weight_norm = F.softmax(attention_pro_comb, dim=-1)
        embs = [weight_norm[:, idx].unsqueeze(1) * F.normalize(embs[idx]) for idx in range(modal_num)]
        joint_emb = torch.cat(embs, dim=1)

        return joint_emb, hidden_states, weight_norm


class MultiModalEncoder(nn.Module):
    """
    entity embedding: (ent_num, input_dim)
    gcn layer: n_units

    """

    def __init__(self, args,
                 ent_num,
                 img_feature_dim,
                 char_feature_dim=None,
                 use_project_head=False,
                 attr_input_dim=1000):
        super(MultiModalEncoder, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim
        img_dim = self.args.img_dim
        name_dim = self.args.name_dim
        char_dim = self.args.char_dim
        dropout = self.args.dropout
        self.ENT_NUM = ent_num
        self.use_project_head = use_project_head

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

        #########################
        # Entity Embedding
        #########################
        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True

        #########################
        # Modal Encoder
        #########################

        self.rel_fc = nn.Linear(1000, attr_dim)
        self.att_fc = nn.Linear(attr_input_dim, attr_dim)
        self.img_fc = nn.Linear(img_feature_dim, img_dim)
        self.name_fc = nn.Linear(300, char_dim)
        self.char_fc = nn.Linear(char_feature_dim, char_dim)
        # self.graph_fc = nn.Linear(self.input_dim, char_dim)

        # structure encoder
        if self.args.structure_encoder == "gcn":
            self.cross_graph_model = GCN(self.n_units[0], self.n_units[1], self.n_units[2],
                                         dropout=self.args.dropout)
        elif self.args.structure_encoder == "gat":
            self.cross_graph_model = GAT(n_units=self.n_units, n_heads=self.n_heads, dropout=args.dropout,
                                         attn_dropout=args.attn_dropout,
                                         instance_normalization=self.args.instance_normalization, diag=True)

        #########################
        # Fusion Encoder
        #########################
        self.fusion = MformerFusion(args, modal_num=self.args.inner_view_num,
                                    with_weight=self.args.with_weight)

    def forward(self,
                input_idx,
                adj,
                img_features=None,
                rel_features=None,
                att_features=None,
                name_features=None,
                char_features=None):

        if self.args.w_gcn:
            gph_emb = self.cross_graph_model(self.entity_emb(input_idx), adj)
        else:
            gph_emb = None
        if self.args.w_img:
            img_emb = self.img_fc(img_features)
        else:
            img_emb = None
        if self.args.w_rel:
            rel_emb = self.rel_fc(rel_features)
        else:
            rel_emb = None
        if self.args.w_attr:
            att_emb = self.att_fc(att_features)
        else:
            att_emb = None
        if self.args.w_name and name_features is not None:
            name_emb = self.name_fc(name_features)
        else:
            name_emb = None
        if self.args.w_char and char_features is not None:
            char_emb = self.char_fc(char_features)
        else:
            char_emb = None

        joint_emb, hidden_states, weight_norm = self.fusion([img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb])

        return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states, weight_norm


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # [8, 8, 3, 256]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        # return x
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        # [8, 3, 8, 256]
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # # [8, 3, 8, 8]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # [8, 3, 8, 8]
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # [8, 8, 768]
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            output_attentions,
        )

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # attention: torch.Size([30355, 5, 4, 4])
        # 5: head
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN["gelu"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = 0
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        if self.config.use_intermediate:
            self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor, output_attentions=False):
        self_attention_outputs = self.attention(
            hidden_states,
            output_attentions=output_attentions,
        )
        if not self.config.use_intermediate:
            return (self_attention_outputs[0], self_attention_outputs[1])

        attention_output = self_attention_outputs[0]
        # if decoder, the last output is tuple of self-attn cache
        outputs = self_attention_outputs[1]
        # present_key_value = self_attention_outputs[-1]
        # torch.Size([30355, 4, 300])
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output, outputs)
        # if decoder, return the attn key/values as the last output
        # outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
        # return attention_output
