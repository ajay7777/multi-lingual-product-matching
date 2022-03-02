# Cross-lingual Product Matching 

This repository contains the code for replicating the experiments submitted as part of the thesis submission.

### Contributors 
These people have contributed to this repository:
 - [@ajay7777](https://github.com/ajay7777)

### Data
The datasets can be requested via mail at ralph@informatik.uni-mannheim.de, but are available for research purposes only. The datasets have to be included in a separate subfolder 'datasets' in the project folder for the experiments.

## Setup
**How to use this Repository**
*Install all of the dependencies from environment.yml
*Before installing, make sure to have [Microsoft Build Tools for C++](https://visualstudio.microsoft.com/de/visual-cpp-build-tools/)  installed for py_entitymatching*

### Settings files

* Individual experiments can be configured using the `.json` settings files. 
* Transformer settings are configured under settings_transformer_pair.json. Different modes of training can be configured in this setting file. e.g. training in cross-lingual mode   can be set using the field - use_cross_lingual_pairs in the settings_transformer_pair.json
* Similary for the LASER and distillation embeddings experiments the settings are in the file settings_laser_pair.json and settings_distillation_pair.json respectively.
* In order to generate the LASER and distillation embeddings run the generate_embeddings.py. However this is not needed as the generated embedded datasets are already shared as part of the data medium in the thesis

### Run Experiments
To run a experiment, make sure to provide the path of the individual `.json` settings file
as input agrument. For instance, to run `settings_transformer_pair.json`, include the argument:

*--i path_to_file\settings_transformer_pair.json*

*To run the transformer model run the transformer_pair.py with the settings file "settings_transformer_pair.json" path as input.
*To run the laser model run the laser_pair.py with the settings file "settings_laser_pair.json" path as input.
*To run the distillation model run the distillation_pair.py with the settings file "settings_laser_pair.json" path as input.

## Cross-lingual and embedding feature configuration
*In order to run the experiments with the cross-lingual config, change the flag "use_cross_lingual_pairs" in the settings file 
*In order to run laser and distillation experiments, use the variable "embeddings_mode" to specify the type of embedding. Two possible values-["abs_diff"] & ["emb_1","emb_2","abs_diff"]
