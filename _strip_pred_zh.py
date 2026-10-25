from pathlib import Path

p = Path(__file__).with_name("prediction_experiment.py")
t = p.read_text(encoding="utf-8").replace("\r\n", "\n")

header_old = t.split("import os", 1)[0]
header_new = '''"""External prediction on dataset/prediction/Additional file 7.csv.

Applies the five base models, majority voting, and SpecEn-Moll. Feature
construction matches utils.data_preprocessing.preprocess: RDKit text
features plus 2-D spectral embeddings; LabelEncoder and StandardScaler
are fit on dataset/train only; unseen object-column values map to -1.

    python prediction_experiment.py
    python prediction_experiment.py --limit 300
"""
'''
t = header_new + "import os" + t.split("import os", 1)[1]

pairs = [
    (
        '    """抽取合法 SMILES。\n\n'
        "    丢弃 RDKit 无法解析的分子：utils/pc_gnn.get_pc_gnn_1 遇到解析失败会 continue\n"
        "    而不追加分子图，若不在这里提前剔除，后续【所有】分子的图与 SMILES 会整体错位。\n"
        '    """',
        '    """Keep RDKit-valid SMILES so graphs stay aligned with the table."""',
    ),
    (
        'raise ValueError(f"源表 {SRC_CSV} 缺少 SMILES 列")',
        'raise ValueError(f"{SRC_CSV} has no SMILES column")',
    ),
    ('dropped.append((i, s, "空值"))', 'dropped.append((i, s, "empty"))'),
    ('dropped.append((i, s, "RDKit 无法解析"))', 'dropped.append((i, s, "invalid SMILES"))'),
    (
        'log(f"[step1] 原始 {len(df)} 行 -> 合法去重后 {len(out)} 条，丢弃 {len(dropped)} 条")',
        'log(f"[step1] {len(df)} rows -> {len(out)} unique valid SMILES, dropped {len(dropped)}")',
    ),
    (
        'log(f"        丢弃行{i} ({why}): {str(s)[:70]}")',
        'log(f"        dropped row {i} ({why}): {str(s)[:70]}")',
    ),
    (
        '    """用 RDKit 复算与训练表同构的 230 列特征。\n\n'
        "    列顺序必须与 dataset/train/text_feature.xlsx 完全一致：\n"
        "        smiles, Mol, MolFormula, InChI, InChIKey, <217 描述符>, <8 指纹>\n"
        "    描述符列表取自当前 RDKit 的 Descriptors._descList（已核对与训练表完全一致）。\n"
        '    """',
        '    """RDKit descriptors and fingerprints aligned with dataset/train/text_feature.xlsx."""',
    ),
    (
        'log(f"[step2] 特征提取失败 {smiles}: {e}")',
        'log(f"[step2] feature extraction failed {smiles}: {e}")',
    ),
    (
        'log(f"[step2] 特征表已生成 shape={df.shape} -> {out}")',
        'log(f"[step2] feature table shape={df.shape} -> {out}")',
    ),
    (
        '    """生成 PyG 分子图并做谱嵌入，得到 feature_0 / feature_1。"""',
        '    """Build molecular graphs and 2-D spectral embeddings (feature_0, feature_1)."""',
    ),
    (
        'log(f"[step3] 谱嵌入完成 shape={emb.shape}")',
        'log(f"[step3] embedding shape={emb.shape}")',
    ),
    (
        '    """从训练表推导模型真正使用的列（去掉 lable），共 230 列。"""',
        '    """Training feature columns excluding the label."""',
    ),
    (
        '    """复现训练期的 scaler / LabelEncoder，并拼装预测集特征矩阵。"""',
        '    """Fit scaler and LabelEncoder on dataset/train; apply to the prediction set."""',
    ),
    (
        'log(f"[step4] 训练期 scaler 已复现（基于 {X.shape[0]} 行 × {X.shape[1]} 列）")',
        'log(f"[step4] scaler fit on train {X.shape[0]} x {X.shape[1]}")',
    ),
    (
        'raise KeyError(f"预测集特征表缺少训练所需的列: {missing}")',
        'raise KeyError(f"prediction table missing training columns: {missing}")',
    ),
    (
        'log(f"[step4] 预测集多出的列（不进入模型）: {extra}")',
        'log(f"[step4] extra columns ignored: {extra}")',
    ),
    (
        '    log(f"[step4] 预测集特征矩阵 scaled={X_scaled.shape} raw={X_raw.shape}"\n'
        '        f"  映射到哨兵值 {UNSEEN_CODE} 的单元格 {unseen_cells} 个")',
        '    log(f"[step4] scaled={X_scaled.shape} raw={X_raw.shape} unseen cells={unseen_cells}")',
    ),
    (
        '    """用已训练模型预测，落盘到 results/prediction/ 下按用途分目录。"""',
        '    """Write base-model, voting, and SpecEn-Moll predictions under results/prediction/."""',
    ),
    (
        'log(f"[step5] {name}: 阳性率={y_pred.mean():.4f} 阳性数={int(y_pred.sum())}/{len(y_pred)}")',
        'log(f"[step5] {name}: positive rate={y_pred.mean():.4f} n={int(y_pred.sum())}/{len(y_pred)}")',
    ),
    (
        'log(f"[step5] SpecEn-Moll 预测跳过: {e}")',
        'log(f"[step5] SpecEn-Moll skipped: {e}")',
    ),
    (
        'log(f"[step5] {agg}: 阳性率={out[col].mean():.4f}")',
        'log(f"[step5] {agg}: positive rate={out[col].mean():.4f}")',
    ),
    (
        '    """无标签条件下的诊断：适用域、类别列未命中率、模型间一致性。"""',
        '    """Applicability domain, unseen object-column rate, and inter-model agreement."""',
    ),
    ('    log("无标签诊断报告")', '    log("unlabeled diagnostics")'),
    (
        '    log(f"  适用域      {len(num_cols)} 个数值描述符越界取值占比 {oob.mean():.2%}；"\n'
        '        f"{oob.any(axis=1).mean():.1%} 的分子至少含 1 个越界描述符")',
        '    log(f"  AD: {len(num_cols)} numeric descriptors, OOB value rate {oob.mean():.2%}; "\n'
        '        f"{oob.any(axis=1).mean():.1%} of molecules have at least one OOB descriptor")',
    ),
    (
        '    log(f"              （训练集内部 30% 留出对 70% 训练的基准越界率 {base_oob.mean():.2%}）")',
        '    log(f"      hold-out vs train baseline OOB rate {base_oob.mean():.2%}")',
    ),
    (
        '    log(f"  类别列      {len(OBJECT_COLS)} 列 × {len(T)} = {len(T)*len(OBJECT_COLS)} 单元格，"\n'
        '        f"其中 {unseen_cells} 个在训练集中未出现过")',
        '    log(f"  object columns: {len(OBJECT_COLS)} x {len(T)} cells, {unseen_cells} unseen in train")',
    ),
    (
        "    log(f\"  一致性      5 个基模型完全一致 {agree['unanimous'].mean():.1%}；\"\n"
        '        f"票数分布 {vote_dist}")',
        "    log(f\"  agreement: unanimous {agree['unanimous'].mean():.1%}; vote dist {vote_dist}\")",
    ),
    (
        'ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 条，用于小规模试跑")',
        'ap.add_argument("--limit", type=int, default=None, help="process the first N rows")',
    ),
    ('log(f"结果输出目录: {RESULT_DIR}")', 'log(f"output directory: {RESULT_DIR}")'),
    ('log(f"小规模试跑：仅处理前 {args.limit} 条")', 'log(f"limit={args.limit}")'),
    (
        'log(f"主结果表已保存: {main_xlsx} shape={out.shape}")',
        'log(f"wrote {main_xlsx} shape={out.shape}")',
    ),
]

ok = 0
for a, b in pairs:
    if a not in t:
        print("MISSING", repr(a[:80]))
    else:
        t = t.replace(a, b)
        ok += 1
p.write_text(t, encoding="utf-8")
print("replaced", ok, "/", len(pairs))
import re
cjk = re.findall(r"[\u4e00-\u9fff]+", t)
print("remaining cjk", set(cjk))
