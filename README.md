# Lingshu-Cell: A Generative Cellular World Model for Transcriptome Modeling toward Virtual Cells

[[Paper]](https://arxiv.org/abs/2603.25240) [[Model Weights]](https://huggingface.co/bibona/lingshu-cell)

## Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

This repository does not include training data or evaluation datasets.

- **VCC (Virtual Cell Challenge)**: CRISPRi perturbation data in H1 cells. Available from [Arc Virtual Cell Atlas](https://github.com/ArcInstitute/arc-virtual-cell-atlas/blob/main/virtual-cell-challenge/README.md).
- **PARSE 10M PBMC**: Single-cell RNA-seq data from [PARSE Biosciences](https://www.parsebiosciences.com/datasets). Must be **aligned to the VCC gene panel (18,080 genes)** -- filter and reorder genes to match `data/vocabs/legacy/*.json`.

Pre-trained model weights and auxiliary files are on [HuggingFace](https://huggingface.co/bibona/lingshu-cell):

| File | Description |
|------|-------------|
| `parse_pbmc_uncond_85M.pth` | PARSE PBMC unconditional model |
| `vcc_perturbation_85M.pth` | VCC gene perturbation model |
| `parse_cytokine_85M.pth` | PARSE cytokine perturbation model |
| `gene_embeddings.npz` | ESM2 gene embeddings (VCC model) |
| `whole_dataset_stringent_clue.csv` | Clue file for biology prior injection (VCC) |

Evaluation requires `cell-eval`: `pip install cell-eval`

## Inference Scripts

### 1. PARSE PBMC Unconditional Generation

```bash
MODEL_PATH=/path/to/parse_pbmc_uncond_85M.pth \
bash scripts/run_infer_and_eval-parse_pbmc_uncond.sh
```

Sampling parameters: `steps=256`, no classifier-free guidance. Set `REAL_DATA=/path/to/pbmc.h5ad` to run evaluation.

### 2. VCC Gene Perturbation (CRISPRi)

```bash
MODEL_PATH=/path/to/vcc_perturbation_85M.pth \
GENE_EMBEDDING_FILE=/path/to/gene_embeddings.npz \
CLUE_CSV=/path/to/whole_dataset_stringent_clue.csv \
bash scripts/run_infer_and_eval-vcc_perturbation.sh
```

Sampling parameters: `steps=3`, `cfg_scale=2`, `cfg_non_targeting`, clue-based biology prior injection. Set `CONTROL_H5AD`, `VCC_TEST_VALIDATION_H5AD`, and `VCC_TEST_BASELINE_DIR` for evaluation.

### 3. PARSE Cytokine Perturbation

```bash
MODEL_PATH=/path/to/parse_cytokine_85M.pth \
DONOR=Donor10 \
bash scripts/run_infer_and_eval-parse_cytokine.sh
```

Sampling parameters: `steps=3`, `cfg_scale=1`, `cfg_general_control`. Test donors: Donor2, Donor6, Donor9, Donor10. Set `PARSE_TEST_DATA_DIR` for evaluation.

## Citation

```bibtex
@article{lingshu-cell,
  title={Lingshu-Cell: A Generative Cellular World Model for Transcriptome Modeling toward Virtual Cells},
  author={Han Zhang and Guo-Hua Yuan and Chaohao Yuan and Tingyang Xu and Tian Bian and Hong Cheng and Wenbing Huang and Deli Zhao and Yu Rong},
  year={2026},
  journal={arXiv preprint arXiv:2603.25240}
}
```

## License

[MIT License](LICENSE). See [NOTICE](NOTICE) for third-party attributions.
