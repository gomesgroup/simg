# SIMG 🧪

## Chemical Representation and Interaction Discovery with Stereoelectronics-Infused Molecular Graphs
Molecular representation is a foundational element in our understanding of the physical world. Its importance ranges from the fundamentals of chemical reactions to the design of new therapies and materials. Previous molecular machine learning models have employed strings, fingerprints, global features, and simple molecular graphs that are inherently information-sparse representations. However, as the complexity of prediction tasks increases, the molecular representation needs to encode higher fidelity information.  This work introduces a novel approach to infusing quantum-chemical-rich information into molecular graphs via stereoelectronic effects. We show that the explicit addition of stereoelectronic interactions significantly improves the performance of molecular machine learning models. Furthermore, stereoelectronics-infused representations can be learned and deployed with a tailored double graph neural network workflow, enabling its application to any downstream molecular machine learning task. Finally, we show that the learned representations allow for facile stereoelectronic evaluation of previously intractable systems, such as entire proteins, opening new avenues of molecular design. 

## Steps to reproduce the results

### Step 1: data preparation

Original NBO data is presented as a large list serialized in json format. The fist step is to convert the json file into
json lines format. This could be done with on a node with large amount of memory:

```bash
python scripts/graph_construction/json_to_jsonl.py --path $JSON_FILE --output $JSONL_FILE
```

Then the NBO data has to be converted into xyz file format. This can be done using the following command:

```bash
python scripts/graph_construction/json_to_xyz_and_nbo.py --path $JSONL_FILE | gzip > $OUTPUT_FILE
```

Multiple files can be converted at the same time (expected time if run in parallel ~ 1 min):

```bash
for i in {1..6}; do python scripts/graph_construction/json_to_xyz_and_nbo.py --path ../data/qm9_nbo7_part$i.json.jsonl | gzip > ../data/qm9_nbo7_part$i.json.jsonl.NBO.gz & done
```

### Step 2: Graph construction

Then we need to generate multiple inputs for various graph operations:

#### Lone pair prediction network

This can be done using `scripts/graph_construction/prepare_LP_prediction_graphs.py` script. See example in "NBO feature prediction network" section.

#### NBO feature prediction network

This can be done by the script located in `scripts/graph_construction/prepare_NBO_prediction_graphs.py` (takes ~ 20 min on a 12-core
machine):

```bash
for i in {1..6}; do python scripts/graph_construction/prepare_NBO_prediction_graphs.py --path ../data/qm9_nbo7_part$i.json.jsonl.NBO.gz --configs scripts/graph_construction.yaml --output_path graphs_$i.pt --mode lps_bonds; done
```

(debug
command `python scripts/graph_construction/prepare_NBO_prediction_graphs.py --path data/test_mol.gz --config configs/graph_construction.yaml --output_path tmp.tmp --mode lps_bonds --debug`)

Then we need to added train/val/test labels to the data point. To enable fair comparison with QM9 baseline, we need to get them from the QM9 dataset (see below). Assuming you have a file with extracted QM9 targets this can be done with the following command:

```bash
for f in *.pt; do python scripts/append_qm9_targets.py --targets_path ../data/qm9/qm9_targets.pkl --graphs_path $f; done
```

The merged files can be merged in a separate folder.

#### Downstream tasks

The first step is to extract corresponding targets (taken from https://github.com/microsoft/tf-gnn-samples):

```bash
python scripts/extract_qm9_targets.py --qm9_path ../data/qm9/ --output_path ../data/qm9/qm9_targets.pkl
```

Then we need to generate the graphs for the downstream tasks in a very similar way as for the NBO prediction network. 

### Step 3: Training the networks
#### Lone pair prediction network

To predict the lone pairs, run the `train.py` located in `experiments/lone_pair_model` (see `python train.py -h` for more details).


#### NBO feature prediction network
To train a model to predict the NBO targets, run the `train.py` located in `experiments/predict_NBO`:

```bash
python train.py --graphs_path $MERGED_GRAPH_PATH --bs 1024 --model_config model_config.yaml --gpus 1
```
The model can be evaluated against the test dataset using the `evaluate.py` script:

```bash
python evaluate.py --graphs_path ../../merged_graphs/ --model_path $CHECKPOINT_PATH
```

#### Downstream tasks

NBO graphs can be evaluated in downstream tasks using the following commands:

```bash
for f in $GRAPHS_PATH/*.pt; do python evaluate_for_downstream.py --graphs_path $f --model_path $CHECKPOINT_PATH --output_path  $(basename ${f}) & done
python test_all_models.py --graphs_path $NEW_GRAPHS_PATH --bs 1024 --parts 12 --model_path model.ckpt --from_NBO
```