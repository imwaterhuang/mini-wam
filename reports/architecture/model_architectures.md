# Mini-WAM 架构对照

依据：`src/mini_wam/models/action_only.py`、`src/mini_wam/models/future_head.py`，核对日期 2026-09-10。两种模型均按当前实现绘制。两者动作路径结构相同，但各自训练后权重可以不同。

图例：蓝色为训练与推理均使用的动作路径；橙色为仅训练的未来预测分支；灰色为冻结的目标编码器；紫色为训练损失。箭头表示前向数据流，虚线表示监督输入。B 表示批次大小，所有形状均包含批次维。MLP（Multilayer Perceptron，多层感知机）；ResNet（Residual Network，残差网络）；ReLU（Rectified Linear Unit，修正线性单元）。

## 1. action-only：当前实现

```mermaid
flowchart TB
  subgraph POLICY["动作路径 · 训练与推理均使用"]
    I["历史图像 o(t−1), o(t)<br/>[B,2,3,96,96]"] --> V["VisualEncoder · 可训练<br/>ResNet-18 · 去分类层<br/>两帧共享编码器 → [B,2,512]"]
    V --> VF["展平 → [B,1024]"]
    P["历史二维位置 p(t−1), p(t)<br/>[B,2,2]"] --> S["StateEncoder<br/>展平为 4 → 64 → 64<br/>每层后 ReLU → [B,64]"]
    VF --> C["拼接 → [B,1088]"]
    S --> C
    C --> H["HistoryFusion<br/>1088 → 512 → 256 · 每层后 ReLU<br/>共享历史表示 h(t)：[B,256]"]
    H --> A["ActionHead<br/>256 → 256 → 32 · 仅第一层后 ReLU<br/>重排 → [B,16,2]"]
    A --> O["预测 16 步动作<br/>â(t), …, â(t+15)<br/>归一化的二维绝对目标坐标"]
  end
  O --> L["动作损失 L_action<br/>Smooth L1 · 只平均有效步<br/>L_total = L_action"]
  GT["真实示范动作 [B,16,2]<br/>action_valid_mask [B,16]"] -.-> L
  classDef shared fill:#eaf3ff,stroke:#3974b9,color:#122c48;
  classDef loss fill:#f3eaff,stroke:#9461b5,color:#44245d;
  class I,V,VF,P,S,C,H,A,O shared;
  class L loss;
```

## 2. future-aware：当前实现

动作路径与上图相同；为避免重复，下图将动作路径中的层细节折叠为模块。

```mermaid
flowchart TB
  subgraph POLICY["相同的动作路径 · 训练与推理均使用"]
    I["历史图像<br/>[B,2,3,96,96]"] --> V["VisualEncoder · 可训练<br/>ResNet-18 → [B,2,512]<br/>展平 → [B,1024]"]
    P["历史二维位置<br/>[B,2,2]"] --> S["StateEncoder<br/>[B,64]"]
    V --> H["拼接 + HistoryFusion<br/>共享历史表示 h(t)<br/>[B,256]"]
    S --> H
    H --> A["ActionHead<br/>预测 16 步动作 [B,16,2]"]
  end
  A --> LA["L_action<br/>带掩码的 Smooth L1"]
  GT["真实示范动作 [B,16,2]<br/>action_valid_mask [B,16]"] -.-> LA
  subgraph FUTURE["新增辅助分支 · 仅训练"]
    AP["真实动作前缀 a(t), …, a(t+3)<br/>action_chunk[:, :4] · [B,4,2]"] --> E["ActionPrefixEncoder<br/>每步共享 MLP：2 → 64 → 64<br/>两层之间加固定正弦余弦位置编码<br/>[B,4,64] → 展平 [B,256]"]
    E --> C["拼接历史表示与动作表示<br/>[B,512]"]
    C --> F["FutureHead · 512 → 1024 → 2048<br/>两层后均 ReLU · 重排 + L2 归一化<br/>预测未来表示 [B,4,512]"]
    Y["真实未来图像 o(t+1), …, o(t+4)<br/>[B,4,3,96,96]"] --> T["FrozenTargetEncoder · 独立 ResNet-18<br/>去分类层 + L2 归一化<br/>目标表示 [B,4,512]<br/>冻结参数 · eval 模式 · 无梯度"]
    F --> LF["L_future<br/>逐步余弦距离 1 − cosine<br/>只平均有效未来步"]
    T -.-> LF
    MASK["future_valid_mask<br/>[B,4]"] -.-> LF
  end
  H --> C
  LA --> TOTAL["L_total = L_action + 0.1 × L_future"]
  LF --> TOTAL
  classDef shared fill:#eaf3ff,stroke:#3974b9,color:#122c48;
  classDef future fill:#fff1df,stroke:#cd8c30,color:#593b14;
  classDef frozen fill:#eef0f3,stroke:#858e9c,color:#394150;
  classDef loss fill:#f3eaff,stroke:#9461b5,color:#44245d;
  class I,V,P,S,H,A shared;
  class AP,E,C,F,Y,MASK future;
  class T frozen;
  class LA,LF,TOTAL loss;
```

## 3. 阅读要点

- 未来分支输入的是数据集中的真实前 4 步动作，不是 ActionHead 的预测动作。
- L_future 的梯度经 FutureHead 和 h(t) 回传到 HistoryFusion、VisualEncoder 与 StateEncoder，也更新 ActionPrefixEncoder；不会经过 ActionHead，也不会更新 FrozenTargetEncoder。
- future-aware 推理时不调用 ActionPrefixEncoder、FutureHead 或 FrozenTargetEncoder。两者均从历史观测直接预测动作，不在推理期进行未来想象或搜索。
- 闭环执行时，预测动作先反归一化并裁剪到合法范围，执行前 4 步，再读取新观测并重新预测。
- 动作前缀的时间编码固定，不参与学习；未来头最后一层之后也使用 ReLU，图中忠实记录当前实现。
- 两者的共同结构与相同初始化用于控制比较；“共享”不表示两个独立模型在训练过程中绑定权重。
