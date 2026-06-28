# PonTED: flexible-linker predictor

PonTED is a per-residue **flexible-linker** predictor for the [CAID challenge](https://caid.idpcentral.org/challenge).
Flexible linkers are the disordered segments that bridge folded domains — a
distinct functional class of disorder in DisProt (`IDPO:0000033`, "flexible
linker"). These give domains the conformational freedom to move relative to one another. PonTED targets that specific signal to help annotate new linkers.

The base method is a small transformer head (~300k params, 1 layer, `d=192`) on top
of a **frozen** protein-language-model embedding, served as a **5-member
ensemble**. It was trained on DisProt flexible-linker annotations (334 proteins, release `2025_12`). Extra training data was processed from TED domains, trimming inter-domain spaces to their disordered core using Boltz structural features. One method variant adds an AlphaFold pLDDT channel and 7 lightweight sequence-biophysics channels

Here we provide three variants, sharing the head architecture:

| Method | Backbone | Extra channels | Runtime inputs | 
|---|---|---|---|
| `PonTED` | ESM-2 650M | none | `--embeddings esm2_path`  |
| `PonTED-XL` | ProtT5 | none | `--embeddings prott5_path` |
| `PonTED-S` | ESM-2 650M | AF2 pLDDT (1) + biophysics (7) | `--embeddings esm2_path --af2-plddt af2_plddt_path` | 


The package is **inference-only**, in two stages:

1. **Precompute inputs** (to be run by host, needs internet and run once): the `precompute/`
   scripts turn a FASTA into per-residue features — pLM embeddings and AlphaFold pLDDT.
2. **Predict** (offline, CPU-only): `predict_caid.py` runs one method over the
   FASTA + precomputed features and writes one `.caid` file per sequence.

## Precompute inputs (to run by hosts)

Create a conda environment with the precompute dependencies, 

```bash
conda create -n ponted-precompute python=3.11 -y && conda activate ponted-precompute
pip install -r precompute/requirements-precompute.txt
```

Then run the processing scripts
```bash
python precompute/compute_esm2.py   --fasta input.fasta --output-dir emb_esm2/    
python precompute/compute_prott5.py --fasta input.fasta --output-dir emb_prott5/  
python precompute/compute_af2.py    --fasta input.fasta --output-dir af2/         
```

Each script writes in `--output-dir` one `<fasta_id>.npy` per sequence. ESM2 is `(L, 1280)`, ProtT5 is `(L, 1024')` and AF2 pLLDT is `(L,)` in `[0,1]`, all float32 values.

The heads were trained against these models:

| Feature | Model / source | dim | loader |
|---|---|---|---|
| ESM-2 embedding | **ESM-2 650M** — `facebook/esm2_t33_650M_UR50D` (33-layer, UniRef50D) | 1280 | `EsmModel` + `AutoTokenizer` |
| ProtT5 embedding | **ProtT5-XL (encoder)** — `Rostlab/prot_t5_xl_half_uniref50-enc` (UniRef50, half precision) | 1024 | `T5EncoderModel` + `T5Tokenizer`; needs `sentencepiece` |
| AF2 pLDDT | **AlphaFold DB** per-residue CA pLDDT (fetched per UniProt accession, ÷100) | 1 | EBI AFDB REST |

Notes: pLM embeddings are the per-residue `last_hidden_state` with CLS/EOS
stripped; pLDDT is the CA value / 100. Sequences longer than 1022 tokens are
embedded in overlapping windows (overlap averaged).

**AF2 needs a UniProt-id list** (only used in `PonTED-S`). The host supplies the mapping via `--id-map` — a TSV of
`fasta_id<TAB>uniprot_acc`, one line per sequence, using `-` (or `nan`) where no
accession is known. Coverage is expected to be **partial**: sequences with no
accession, or whose accession has no AlphaFold model get no pLDDT file.

## Prediction

The image is published on Docker Hub. Pull it once:

```bash
docker pull lbugnon/ponted:caid
```

```bash
# ESM-2 base
docker run --rm --network none \
  -v $PWD/input.fasta:/data/input.fasta:ro \
  -v $PWD/emb_esm2:/data/embeddings:ro \
  -v $PWD/predictions/PonTED:/data/output \
  lbugnon/ponted:caid --method PonTED \
  --fasta /data/input.fasta --embeddings /data/embeddings \
  --out /data/output --threads 8
```

```bash
# ESM-2 structure-aware variant
docker run --rm --network none \
  -v $PWD/input.fasta:/data/input.fasta:ro \
  -v $PWD/emb_esm2:/data/embeddings:ro \
  -v $PWD/af2:/data/af2:ro \
  -v $PWD/predictions/PonTED-S:/data/output \
  lbugnon/ponted:caid --method PonTED-S \
  --fasta /data/input.fasta --embeddings /data/embeddings \
  --af2-plddt /data/af2 --out /data/output --threads 8
```

```bash
# ProtT5 variant
docker run --rm --network none \
  -v $PWD/input.fasta:/data/input.fasta:ro \
  -v $PWD/emb_prott5:/data/embeddings:ro \
  -v $PWD/predictions/PonTED-XL:/data/output \
  lbugnon/ponted:caid --method PonTED-XL \
  --fasta /data/input.fasta --embeddings /data/embeddings \
  --out /data/output --threads 8
```

### Input arguments

| Flag | Required | Description |
|---|---|---|
| `--method` | yes | Method name |
| `--fasta` | yes | Input FASTA format |
| `--embeddings` | yes | Directory with per-id `<id>.npy`. |
| `--out` | yes | Output directory. |
| `--af2-plddt` | for af2 methods | Directory of `<id>.npy` pLDDT in `[0,1]`. |
| `--threads` | no |  CPU threads (default 8). |

### Output

```
predictions/
  <id>.caid       # position \t residue \t score \t binary_state
  timings.csv     # per-sequence wall time
```

Each `.caid` is 4-column (position, residue, score, binary state), per the CAID
output format. The score is the linker ranking signal; the binary state is
`score >= binary_threshold` (set per method in `method.yaml`).
