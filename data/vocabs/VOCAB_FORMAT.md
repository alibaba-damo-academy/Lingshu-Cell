# Dense 表示码本格式规范

本文档定义 general tokenizer (`DenseGene1TokTokenizer`) 使用的 vocab JSON 文件格式。

## 必含字段

### `special_token_vocab`

特殊 token 名称到 token ID 的映射。

```json
{
  "<unk>": 0,
  "<s>": 1,
  "</s>": 2,
  "<cond>": 3,
  "<unk_gene>": 4
}
```

**必须包含**：`</s>`（EOS / padding）、`<unk_gene>`（未知基因回退 token）。

bookend token（如 `<cond>`）由 `ConditionFormat.bookend_token` 指定，也必须在此字段中。

### `count_to_token_id_info`

基因表达计数的编码参数。

```json
{
  "min_count": 0,
  "max_count": 9999,
  "min_token_id": 5,
  "max_token_id": 599,
  "overflow_token_id": 599,
  "count_mapping_rules": [
    {
      "min_count": 0,
      "max_count": 99,
      "min_token_id": 5,
      "max_token_id": 104,
      "mapping_method": "identity"
    },
    {
      "min_count": 100,
      "max_count": 9999,
      "min_token_id": 105,
      "max_token_id": 599,
      "mapping_method": "2_important_digits"
    }
  ]
}
```

| 子字段 | 类型 | 说明 |
|--------|------|------|
| `min_count` | int | 最小可表示计数（通常 0） |
| `max_count` | int | 最大可表示计数，超过则 overflow |
| `min_token_id` | int | 对应 `min_count` 的 token ID |
| `max_token_id` | int | 计数区间的最大 token ID |
| `overflow_token_id` | int | count > max_count 时使用的 token ID |
| `count_mapping_rules` | list | 有序映射规则列表（见下） |

每条规则覆盖一个计数区间，`mapping_method` 取值：

- `"identity"`：线性 1:1 映射，`token_id = min_token_id + (count - min_count)`
- `"2_important_digits"`：保留两位有效数字的有损压缩

### `max_gene_count`

int — 基因 panel 大小（即 dense 序列中表达 token 的数量）。

### `gene_names_ordered`

list[str] — 有序基因名列表。列表索引 = dense position。

```json
"gene_names_ordered": ["BRCA1", "TP53", "MYC", ...]
```

## 条件码本字段

以 `_token_vocab` 结尾的字段，值为 `dict[str, int]`（标签 → token ID）。

由 `ConditionFormat` 中各 slot 的 `codebook_key` 引用。tokenizer 只加载被 slot 引用的码本。

```json
"cell_type_token_vocab": {"H1": 105, "HepG2": 106, "K562": 107},
"donor_token_vocab": {"donor_1": 200, "donor_2": 201},
"cytokine_token_vocab": {"IFNg": 300, "TNFa": 301}
```

## 可选字段

### `special_control_vocab`

控制样本专用标签到 token ID 的映射。第一个 key 作为 `control_label`。

```json
"special_control_vocab": {"non-targeting": 107}
```

**兼容旧码本**：旧 vocab 文件中此字段名为 `special_target_gene_vocab`，tokenizer 按优先级查找 `special_control_vocab` → `special_target_gene_vocab`。

### `gene_index_offset`

int — gene-index 类型 slot 的 token ID 偏移量。`token_id = gene_index_offset + dense_idx`。

**兼容旧码本**：旧 vocab 文件中此字段放在 `target_gene_to_token_id_info.target_gene_token_id_offset` 下（见下方兼容节）。新码本建议放在顶层或对应的 codebook 字段内。

## 旧码本兼容（VCC 格式）

VCC 格式的 vocab 文件将 gene panel 信息嵌套在 `target_gene_to_token_id_info` 字段下：

```json
"target_gene_to_token_id_info": {
  "target_gene_token_id_offset": 108,
  "max_gene_count": 5000,
  "gene_names_ordered": ["BRCA1", "TP53", ...],
  "target_gene_encoding": "1token"
}
```

General tokenizer 的兼容策略：

1. 当 `ConditionFormat` 中有 `gene_index` slot 且其 `codebook_key` 指向此字段时，从中读取 `target_gene_token_id_offset`、`max_gene_count`、`gene_names_ordered`
2. 当无 `gene_index` slot 时，从 vocab 顶层读取 `max_gene_count` 和 `gene_names_ordered`（fallback 路径）

## 现有文件

| 文件 | 说明 |
|------|------|
| `data/vocabs/parse_gene1tok.json` | PARSE 配置码本（无 gene_index slot，gene panel 在顶层） |
| `data/vocabs/legacy/tokenizer_vcc_cond_dense_gene1tok.json` | VCC 码本（旧格式） |
| `data/vocabs/legacy/tokenizer_vcc_cond_dense_gene1tok_parse.json` | VCC-PARSE 码本（旧格式） |
