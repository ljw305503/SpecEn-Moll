import re
from pathlib import Path

p = Path(__file__).with_name("prediction_experiment.py")
t = p.read_text(encoding="utf-8").replace("\r\n", "\n")
# untokenize inserted a blank line after every line
lines = t.split("\n")
if sum(1 for i in range(0, min(40, len(lines)), 2) if lines[i] == "") > 10:
    t = "\n".join(lines[i] for i in range(len(lines)) if i % 2 == 0 or lines[i] != "")

# leftover Chinese blocks
t = re.sub(
    r'    """抽取合法 SMILES。.*?"""',
    '    """Keep RDKit-valid SMILES so graphs stay aligned with the table."""',
    t,
    count=1,
    flags=re.S,
)
t = re.sub(
    r'    """用 RDKit 复算与训练表同构的 230 列特征。.*?"""',
    '    """RDKit descriptors and fingerprints aligned with dataset/train/text_feature.xlsx."""',
    t,
    count=1,
    flags=re.S,
)
t = t.replace(
    '    log(f"[step4] 预测集特征矩阵 scaled={X_scaled.shape} raw={X_raw.shape}"\n'
    '        f"  映射到哨兵值 {UNSEEN_CODE} 的单元格 {unseen_cells} 个")',
    '    log(f"[step4] scaled={X_scaled.shape} raw={X_raw.shape} unseen cells={unseen_cells}")',
)
t = t.replace(
    '    log(f"  适用域      {len(num_cols)} 个数值描述符越界取值占比 {oob.mean():.2%}；"\n'
    '        f"{oob.any(axis=1).mean():.1%} 的分子至少含 1 个越界描述符")',
    '    log(f"  AD: {len(num_cols)} numeric descriptors, OOB value rate {oob.mean():.2%}; "\n'
    '        f"{oob.any(axis=1).mean():.1%} of molecules have at least one OOB descriptor")',
)
t = t.replace(
    '    log(f"  类别列      {len(OBJECT_COLS)} 列 × {len(T)} = {len(T)*len(OBJECT_COLS)} 单元格，"\n'
    '        f"其中 {unseen_cells} 个在训练集中未出现过")',
    '    log(f"  object columns: {len(OBJECT_COLS)} x {len(T)} cells, {unseen_cells} unseen in train")',
)
t = t.replace(
    "    log(f\"  一致性      5 个基模型完全一致 {agree['unanimous'].mean():.1%}；\"\n"
    '        f"票数分布 {vote_dist}")',
    "    log(f\"  agreement: unanimous {agree['unanimous'].mean():.1%}; vote dist {vote_dist}\")",
)

p.write_text(t, encoding="utf-8")
left = set(re.findall(r"[\u4e00-\u9fff]+", t))
print("lines", t.count(chr(10)), "cjk", left)
