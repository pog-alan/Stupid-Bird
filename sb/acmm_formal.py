from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class FormalObject:
    symbol: str
    name: str
    domain: str
    definition: str
    invariants: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "domain": self.domain,
            "definition": self.definition,
            "invariants": list(self.invariants),
        }


@dataclass(frozen=True)
class FormalMap:
    name: str
    signature: str
    equation: str
    role: str
    assumptions: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "signature": self.signature,
            "equation": self.equation,
            "role": self.role,
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class FormalMetric:
    name: str
    equation: str
    meaning: str
    target: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "equation": self.equation,
            "meaning": self.meaning,
            "target": self.target,
        }


@dataclass(frozen=True)
class FormalSection:
    title: str
    goal: str
    objects: Tuple[FormalObject, ...] = ()
    maps: Tuple[FormalMap, ...] = ()
    metrics: Tuple[FormalMetric, ...] = ()
    notes: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {
            "title": self.title,
            "goal": self.goal,
            "objects": [item.as_dict() for item in self.objects],
            "maps": [item.as_dict() for item in self.maps],
            "metrics": [item.as_dict() for item in self.metrics],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ACMMFormalSpec:
    name: str = "Affective-Causal Memory Model Formal Specification"
    short_name: str = "ACMM"
    objective: str = "用对象化状态、因果先验、情绪监督和分层记忆缩小学习与复核空间。"
    sections: Tuple[FormalSection, ...] = field(default_factory=tuple)

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "short_name": self.short_name,
            "objective": self.objective,
            "global_loop": "x_t -> O_t -> z_t -> G_t -> e_t -> g_t -> (M_{t+1}, G_{t+1}, theta_{t+1}, a_t)",
            "sections": [section.as_dict() for section in self.sections],
            "acceptance_criteria": self.acceptance_criteria(),
            "implementation_mapping": self.implementation_mapping(),
        }

    def validate(self) -> None:
        required = {
            "输入与观测空间",
            "对象化与关系图",
            "世界状态编码",
            "因果图与预测",
            "分层记忆",
            "情绪监督向量",
            "情绪门控",
            "动作与复核策略",
            "训练目标与参数更新",
            "真实语料标注与 A/B 验证",
        }
        titles = {section.title for section in self.sections}
        missing = required - titles
        if missing:
            raise ValueError(f"ACMM 形式化规格缺少章节：{sorted(missing)}")

    def acceptance_criteria(self) -> List[str]:
        return [
            "所有输入必须先映射到对象集合 O_t 与关系集合 R_t，不能直接把原始 token 当成最终状态。",
            "世界状态 z_t 必须至少包含对象、属性、状态、关系和时间索引。",
            "情绪向量 e_t 必须可由预测误差、不确定性、新颖度、风险、价值、冲突和信息增益计算或监督得到。",
            "记忆写入、模型更新、人工复核、规则更新和风险预警必须由门控函数 g(e_t) 控制。",
            "训练目标必须同时报告任务损失、预测损失、因果一致性损失、记忆损失、规则损失和反馈损失。",
            "真实数据报告必须区分 human label 与 weak label；弱标签实验只能证明流程可跑，不能证明真实收益。",
            "A/B 验证必须至少包含 baseline、memory、causal、acmm 和 random 五条路径。",
        ]

    def implementation_mapping(self) -> Dict[str, str]:
        return {
            "对象/关系/状态": "sb/acmm.py:ObjectState, RelationState, WorldState",
            "因果图": "sb/acmm.py:CausalRule, CausalGraph",
            "分层记忆": "sb/acmm.py:LayeredMemoryStore, MemoryItem",
            "情绪向量与门控": "sb/acmm.py:ACMMEmotionVector, compute_gates",
            "认知循环": "sb/acmm.py:AffectiveCausalMemoryModel.cognitive_step",
            "文本对象化": "sb/acmm_text.py:build_text_observation",
            "Chinese-C4 弱监督评测": "examples/v02_acmm_chinese_c4_eval.py",
            "人工标注": "examples/v02_acmm_chinese_c4_label_tool.py",
            "A/B 验证": "examples/v02_acmm_chinese_c4_ab_eval.py",
        }

    def to_markdown(self) -> str:
        lines = [f"# {self.short_name} 形式化数学对象规格", "", self.objective, ""]
        lines.append("```text")
        lines.append("x_t -> O_t -> z_t -> G_t -> e_t -> g_t -> (M_{t+1}, G_{t+1}, theta_{t+1}, a_t)")
        lines.append("```")
        lines.append("")
        for section in self.sections:
            lines.extend([f"## {section.title}", "", section.goal, ""])
            if section.objects:
                lines.append("### 数学对象")
                lines.append("")
                for item in section.objects:
                    lines.append(f"- `{item.symbol}`：{item.name}")
                    lines.append(f"  定义域：`{item.domain}`")
                    lines.append(f"  定义：{item.definition}")
                    for invariant in item.invariants:
                        lines.append(f"  不变量：{invariant}")
                lines.append("")
            if section.maps:
                lines.append("### 计算映射")
                lines.append("")
                for item in section.maps:
                    lines.append(f"- `{item.name}`")
                    lines.append(f"  签名：`{item.signature}`")
                    lines.append(f"  公式：`{item.equation}`")
                    lines.append(f"  作用：{item.role}")
                    for assumption in item.assumptions:
                        lines.append(f"  假设：{assumption}")
                lines.append("")
            if section.metrics:
                lines.append("### 指标")
                lines.append("")
                for item in section.metrics:
                    lines.append(f"- `{item.name}`：`{item.equation}`")
                    lines.append(f"  含义：{item.meaning}")
                    lines.append(f"  目标：{item.target}")
                lines.append("")
            if section.notes:
                lines.append("### 备注")
                lines.append("")
                for note in section.notes:
                    lines.append(f"- {note}")
                lines.append("")
        lines.append("## 验收标准")
        lines.append("")
        for item in self.acceptance_criteria():
            lines.append(f"- {item}")
        lines.append("")
        return "\n".join(lines)


def build_acmm_formal_spec() -> ACMMFormalSpec:
    return ACMMFormalSpec(sections=_sections())


def acmm_formal_spec_dict() -> Dict[str, object]:
    spec = build_acmm_formal_spec()
    spec.validate()
    return spec.as_dict()


def acmm_formal_markdown() -> str:
    spec = build_acmm_formal_spec()
    spec.validate()
    return spec.to_markdown()


def _sections() -> Tuple[FormalSection, ...]:
    return (
        FormalSection(
            title="输入与观测空间",
            goal="定义系统处理的原始输入、任务目标、标签和外部动作空间。",
            objects=(
                FormalObject(
                    "t in N",
                    "离散时间索引",
                    "t = 0,1,2,...",
                    "每次输入、预测、记忆写入和参数更新都发生在一个离散步。",
                    ("所有状态变量必须带有 t 或可追溯到某个 t。",),
                ),
                FormalObject(
                    "x_t",
                    "原始观测",
                    "X_text union X_image union X_sensor union X_gis",
                    "第 t 步输入，可为文本、图像、遥感影像、传感器或 GIS 数据。",
                    ("x_t 不直接作为最终推理状态，必须先对象化。",),
                ),
                FormalObject(
                    "y_t",
                    "任务标签或输出目标",
                    "Y",
                    "任务结果，例如普通文本、弱风险线索、高风险/需复核，或分割/分类标签。",
                ),
                FormalObject(
                    "a_t",
                    "信息动作",
                    "{retrieve, review, write, update_rule, alert, train, ask}",
                    "非具身系统的动作是信息操作，而不是物理动作。",
                ),
            ),
            maps=(
                FormalMap(
                    "采样过程",
                    "D = {(x_i, y_i, meta_i)}_{i=1..n}",
                    "x_i ~ P_data(X), y_i ~ P_label(Y | x_i)",
                    "把真实语料、合成样本或人工标注集统一成数据集对象。",
                    ("weak label 与 human label 必须分开记录。",),
                ),
            ),
        ),
        FormalSection(
            title="对象化与关系图",
            goal="把原始输入从 token/像素空间映射到对象、属性、状态和关系空间。",
            objects=(
                FormalObject(
                    "O_t = {o_i^t}_{i=1..n_t}",
                    "对象集合",
                    "finite set of object states",
                    "每个对象是 o_i=(id_i,type_i,attr_i,loc_i,state_i,conf_i,time_i)。",
                    ("0 <= conf_i <= 1", "对象 id 在同一 z_t 内应唯一。"),
                ),
                FormalObject(
                    "A_i^t",
                    "对象属性向量",
                    "R^{d_a} or sparse key-value map",
                    "描述颜色、形状、语义属性、关键词命中、遥感指数等。",
                ),
                FormalObject(
                    "R_t = {(o_i,r_ij,o_j,c_ij)}",
                    "关系集合",
                    "O_t × Rel × O_t × [0,1]",
                    "对象之间的空间、语义、时序或文本共现关系。",
                ),
            ),
            maps=(
                FormalMap(
                    "对象化函数",
                    "F_obj: X -> P(O)",
                    "O_t = F_obj(x_t; theta_obj)",
                    "从文本、图像、传感器或规则中抽取对象候选。",
                ),
                FormalMap(
                    "关系抽取函数",
                    "F_rel: O_t × X -> P(R)",
                    "R_t = F_rel(O_t, x_t; theta_rel)",
                    "从对象位置、共现、语法、空间邻接或时序变化中抽取关系。",
                ),
            ),
        ),
        FormalSection(
            title="世界状态编码",
            goal="把对象图转成可比较、可预测、可写入记忆的世界状态。",
            objects=(
                FormalObject(
                    "z_t",
                    "世界状态",
                    "Z = Graph(O,R,A,S,time)",
                    "z_t=(O_t,R_t,meta_t)，是对象、属性、状态和关系组成的图结构。",
                    ("z_t 必须保留可解释对象结构，不能只剩一个黑箱向量。",),
                ),
                FormalObject(
                    "phi(z_t)",
                    "状态嵌入",
                    "R^d",
                    "用于相似度检索、预测误差和新颖度计算的向量化表示。",
                    ("||phi(z_t)||_2 = 1 或显式记录未归一化。",),
                ),
            ),
            maps=(
                FormalMap(
                    "状态编码",
                    "F_state: P(O) × P(R) × M -> Z",
                    "z_t = F_state(O_t, R_t, Retrieve(M_t, O_t, R_t))",
                    "将当前对象图与记忆召回融合成世界状态。",
                ),
                FormalMap(
                    "图嵌入",
                    "phi: Z -> R^d",
                    "phi(z_t)=Normalize(sum_i Emb(o_i)+sum_j Emb(r_j))",
                    "为召回、预测和度量提供统一向量接口。",
                ),
            ),
        ),
        FormalSection(
            title="因果图与预测",
            goal="定义先验机制、因果边和下一状态预测，提供 Surprise 与 Conflict 的来源。",
            objects=(
                FormalObject(
                    "G_t=(V_t,E_t,W_t)",
                    "因果图",
                    "directed weighted graph",
                    "V_t 是状态变量，E_t 是因果边，W_t 是边置信度或机制强度。",
                    ("因果边有方向，不能退化成无向相关边。",),
                ),
                FormalObject(
                    "epsilon_t",
                    "扰动项",
                    "noise space Epsilon",
                    "未观测因素、季节波动、噪声或采样误差。",
                ),
            ),
            maps=(
                FormalMap(
                    "结构因果预测",
                    "F_cau: Z × A × G -> Z",
                    "hat_z_{t+1}=F_cau(z_t,a_t,G_t)",
                    "基于当前状态、信息动作和因果图预测下一状态。",
                    ("可用规则、神经网络或混合机制实现。",),
                ),
                FormalMap(
                    "反事实预测",
                    "F_cf: Z × do(A=a') × G -> Z",
                    "z_{t+1}^{cf}=F_cau(z_t,do(a'),G_t)",
                    "用于判断如果采取不同信息动作，风险和不确定性是否下降。",
                ),
            ),
            metrics=(
                FormalMetric(
                    "预测误差",
                    "delta_t = d(z_t, hat_z_t)",
                    "当前状态与上一轮预测的差异。",
                    "delta_t 越大，Surprise 越大。",
                ),
                FormalMetric(
                    "规则冲突",
                    "C_t = violation(z_t,G_t,K_t)",
                    "当前状态违反因果图或规则库的程度。",
                    "冲突越高，update_rule/review 门控应越高。",
                ),
            ),
        ),
        FormalSection(
            title="分层记忆",
            goal="把经验分为情景、语义、因果、规则和反例记忆，避免所有样本等价写入。",
            objects=(
                FormalObject(
                    "M_t={M_epi,M_sem,M_cau,M_rule,M_neg}",
                    "分层记忆",
                    "tuple of memory stores",
                    "情景记忆、语义记忆、因果记忆、规则记忆和反例记忆的集合。",
                    ("每个记忆项必须带有来源、时间、情绪向量和标签来源。",),
                ),
                FormalObject(
                    "m_i",
                    "记忆项",
                    "R^d × Z × Y × E × Type × Meta",
                    "m_i=(phi_i,z_i,y_i,e_i,type_i,meta_i)。",
                ),
            ),
            maps=(
                FormalMap(
                    "记忆召回",
                    "Retrieve: M × Z -> TopK(M)",
                    "R_t^M = TopK_{m_i in M}(sim(phi(z_t), phi_i) + beta type_bonus_i)",
                    "从分层记忆中召回与当前状态最相关的经验。",
                ),
                FormalMap(
                    "记忆写入",
                    "Write: M × Z × Y × E × GATE -> M",
                    "M_{t+1}=M_t union {m_t} if g_write(e_t)>tau_write",
                    "只有超过情绪门控阈值的经验写入长期记忆。",
                ),
            ),
            metrics=(
                FormalMetric(
                    "新颖度",
                    "N_t = 1 - max_{m_i in M_t} sim(phi(z_t), phi_i)",
                    "当前状态与已有记忆最相似样本的距离。",
                    "写入相似记忆后，重复样本的 Novelty 应下降。",
                ),
            ),
        ),
        FormalSection(
            title="情绪监督向量",
            goal="把预测误差、不确定性、新颖度、风险、价值、冲突和信息增益压缩成监督变量。",
            objects=(
                FormalObject(
                    "e_t",
                    "ACMM 七维情绪监督向量",
                    "[0,1]^7",
                    "e_t=(S_t,U_t,N_t,R_t,V_t,C_t,Q_t)。",
                    ("所有分量归一到 [0,1]。", "e_t 是监督/调制变量，不表示主观体验。"),
                ),
                FormalObject("S_t", "Surprise", "[0,1]", "预测误差归一化值。"),
                FormalObject("U_t", "Uncertainty", "[0,1]", "输出分布熵归一化值。"),
                FormalObject("N_t", "Novelty", "[0,1]", "与已有记忆的最小相似距离。"),
                FormalObject("R_t", "Risk", "[0,1]", "错判代价或风险强度。"),
                FormalObject("V_t", "Value", "[0,1]", "当前样本对任务目标的重要性。"),
                FormalObject("C_t", "Conflict", "[0,1]", "规则/因果冲突程度。"),
                FormalObject("Q_t", "Curiosity", "[0,1]", "信息增益潜力。"),
            ),
            maps=(
                FormalMap(
                    "情绪评价",
                    "Emotion: Z × Z_hat × P(Y) × M × G × Goal -> [0,1]^7",
                    "e_t = [d(z_t,hat_z_t), H(p_t), 1-max sim, E Cost, Utility, violation, IG]",
                    "生成训练、推理和预测共享的情绪监督。",
                ),
                FormalMap(
                    "信息增益近似",
                    "IG: Z × D -> [0,1]",
                    "Q_t approx U_t * N_t * V_t",
                    "在没有完整贝叶斯后验时，用不确定性、新颖度和价值近似好奇度。",
                ),
            ),
        ),
        FormalSection(
            title="情绪门控",
            goal="用情绪监督控制学习强度、记忆写入、人工复核、规则更新和风险预警。",
            objects=(
                FormalObject(
                    "g_t",
                    "门控向量",
                    "[0,1]^5",
                    "g_t=(g_write,g_update,g_review,g_rule,g_alert)。",
                    ("门控值不能直接作为标签，只能作为调度/权重。",),
                ),
            ),
            maps=(
                FormalMap(
                    "门控函数",
                    "Gate: [0,1]^7 -> [0,1]^5",
                    "g_t = sigma(W_g e_t + b_g)",
                    "把情绪监督映射为系统动作强度。",
                ),
                FormalMap(
                    "样本权重",
                    "Weight: [0,1]^5 -> R_+",
                    "w_t = 1 + alpha_update*g_update + alpha_review*g_review",
                    "把门控接入训练采样权重或损失权重。",
                ),
            ),
        ),
        FormalSection(
            title="动作与复核策略",
            goal="定义不具身系统的信息动作和复核队列选择问题。",
            objects=(
                FormalObject(
                    "pi(a_t | z_t,e_t,M_t,G_t)",
                    "动作策略",
                    "probability distribution over A",
                    "决定查询、写入、训练、复核、更新规则或预警。",
                ),
                FormalObject(
                    "B",
                    "复核预算",
                    "integer budget",
                    "每轮最多允许人工审核的样本数量。",
                ),
            ),
            maps=(
                FormalMap(
                    "复核排序",
                    "Rank: D_eval -> permutation",
                    "score_i = max(g_review_i,g_alert_i,g_rule_i)",
                    "在给定预算下选择最值得复核的样本。",
                ),
            ),
            metrics=(
                FormalMetric(
                    "Precision@B",
                    "P@B = |TopB ∩ Positive| / B",
                    "复核队列中真正高风险/需复核样本比例。",
                    "越高越好。",
                ),
                FormalMetric(
                    "Recall@B",
                    "R@B = |TopB ∩ Positive| / |Positive|",
                    "在预算内捕获的高风险样本比例。",
                    "越高越好。",
                ),
            ),
        ),
        FormalSection(
            title="训练目标与参数更新",
            goal="把任务损失、预测损失、因果一致性、记忆一致性、规则约束和反馈损失统一起来。",
            objects=(
                FormalObject(
                    "theta",
                    "可训练参数",
                    "parameter space Θ",
                    "对象抽取器、状态编码器、因果预测器、分类器、门控器或 SB-Core 主干参数。",
                ),
                FormalObject(
                    "lambda",
                    "损失权重",
                    "R_+^k",
                    "控制各项监督在总目标中的比例。",
                ),
            ),
            maps=(
                FormalMap(
                    "总损失",
                    "L: Θ × D -> R_+",
                    "L=L_task+λ1L_pred+λ2L_causal+λ3L_memory+λ4L_rule+λ5L_feedback+λ6L_emotion",
                    "训练 ACMM/SB-Core 时统一使用的复合目标。",
                ),
                FormalMap(
                    "情绪门控参数更新",
                    "Update: Θ × E × grad L -> Θ",
                    "theta_{t+1}=theta_t-eta*w(e_t)*grad_theta L_t",
                    "不是所有样本平等影响模型，学习强度由情绪监督调制。",
                ),
            ),
            metrics=(
                FormalMetric("任务损失", "L_task=CE(y_t,hat_y_t)", "分类/生成/分割主任务误差。", "越低越好。"),
                FormalMetric("预测损失", "L_pred=d(z_{t+1},hat_z_{t+1})", "时序或因果预测误差。", "越低越好。"),
                FormalMetric("反馈损失", "L_feedback=CE(y_human,hat_y_t)", "人工复核标签监督。", "越低越好。"),
            ),
        ),
        FormalSection(
            title="真实语料标注与 A/B 验证",
            goal="定义从 Chinese-C4 到标注集，再到 baseline/memory/causal/acmm 对照的评测对象。",
            objects=(
                FormalObject(
                    "D_c4",
                    "Chinese-C4 样本集",
                    "JSONL text rows",
                    "真实中文网页文本样本，用于弱监督筛选和人工标注。",
                ),
                FormalObject(
                    "D_label",
                    "人工标注集",
                    "{(x_i,y_i^human,meta_i)}",
                    "人工标注普通文本、弱风险线索、高风险/需复核或不确定。",
                    ("A/B 结论必须优先基于 D_label。",),
                ),
                FormalObject(
                    "D_weak",
                    "弱标签集",
                    "{(x_i,y_i^weak,rule_i)}",
                    "由关键词和规则产生，只能作为 smoke 或候选采样辅助。",
                ),
            ),
            maps=(
                FormalMap(
                    "A/B 排序器",
                    "S_k: D_label -> R, k in {baseline,memory,causal,acmm,random}",
                    "rank_k = argsort_i S_k(x_i)",
                    "同一标注集上比较不同机制的复核排序能力。",
                ),
            ),
            metrics=(
                FormalMetric(
                    "相对提升",
                    "Lift_k = Precision@B_k / Precision@B_random",
                    "某方法相对随机复核的精度提升。",
                    "大于 1 表示优于随机。",
                ),
                FormalMetric(
                    "弱标签警戒",
                    "source_ratio = |human labels| / |all eval labels|",
                    "报告中人工标签占比。",
                    "正式结论要求 source_ratio 足够高。",
                ),
            ),
            notes=(
                "如果 allow_weak_labels=True，报告只能说明流程可运行。",
                "如果 baseline 使用了生成弱标签的同一规则，它在弱标签评测中会天然偏强。",
            ),
        ),
    )
