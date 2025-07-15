BlockGPT: Readme to follow

Structure of the repo


configs:
        Evaluations: evaluation configs where the paths to the checkpoints of models to be evaluated are placed.
        GPT: all blockGPT configs including nowcastingGPT. 
        config_vqgan: vqgan config which is used in the study (vae and vqgan perturb are other encoders which can be benchmarked)
        config_blockGPTBackbone: this is the config for the blockGPT backbone used in Diffcast

dataset:
        get_datasets.py: this is the main dataset script where the paths to the dataset must be set. It calls the dataloaders in the other scripts.
        dataset_knmi and dataset_sevir are the corresponding dataloaders.

job_scripts:
        contain job scripts to train the encoders, blockGPT, diffcast + phydnet and diffcast + blockgpt and nowcsatingGPT.
        note that these have only been used for snellius. For quechua, I run the .py (using the same command as in  the job script) in a screen session (alternate to nohup)

models:
        blockGPT: contains the GPT class used. Note that nowcastingGPT also uses this class but with a simple config change (is_block is set to False). 

Results: 

        Evaluations: stores prediction map plots as well as .pkl files of metrics which are outputs of evaluate.py. It also writes a .txt file of the names of the models that have been evaluated.

        FinalPickledResults: contains pickle results of models that have been evaluated for the thesis.

utils: utilites used by training code.

scripts:
    train_gpt.py- to train blockGPT or nowcastingGPT
    train_diffcast.py- to train diffcast with either phydnet or blockGPT
    train_encoder.py- to train vqgan, vae or vqgan with perturb
    evaluate.py- evaluates all models
    plot_metrics.ipynb- contains the plotting functions used in my thesis. It retrieves the pickle files in FinalPickledResults and plots them

    






Instructions to create the environment

Step 1: conda env create -f blockgpt_env.yml
Step 2 (only if you get the error importing cached_download in dynamic_utils.py when you run training):
        Replace this line in dynamic_utils.py (path to the file will be mentioned in the error log)
        from huggingface_hub import cached_download, hf_hub_download, model_info
        with 
        from huggingface_hub import hf_hub_download, model_info
Step 3: pip install imageio==2.33.0 --no-deps 
Step 4: pip install ema-pytorch

Note that all paths in the github repository correspond to quechua paths. A folder on snellius in the name of 'blockGPT' is a copy of this repository but with the paths adjusted everywhere for snellius.

Arranging the Dataset

KNMI: The dataset used in this study has been curated from the raw KNMI precipitation dataset.
Curation steps will be explained in the appendix. Assuming that it has already been curated, you must change the locations in get_datases.py located in the dataset folder to point to the curated dataset.

Note that KNMI 30 minutes and KNMI 5 minutes are inherently different sources. So please remember to change both paths.

SEVIR: SEVIR comes directly available with a dataloader. Place the raw dataset's path in get_datasets.py


Training the Encoder

The encoder used in the study is a vqgan. Simply run the corresponding job script. Note that quechua does not need job scripts. You can simply run the actual command (found within the job script) in a screen session (or any other tool of your choice like nohup).


Training the GPT model

Again, simply run the corresponding job script. Make sure that you have the checkpoints of the encoder passed correctly in the args.
On quechua, and just for the GPT model, training works only on a single GPU. The command to run single GPU training (again needed only on quechua) is also in the job script (commented).

Training Diffcast+PhyDnet

Run the corresponding job script. Note that there is no config because the authors of diffcast have very simple config arguments which are defined in the training script. I retain this black box and thus do not change anything.

Training Diffcast+BlockGPT

The correspondning job script. But now note that there is a model config passed as well. I have tested it only on 30 minutes SEVIR, but you can easily change the config and training command in the job script. Make sure that you change the encoder and GPT paths correctly in the config.

Evaluation

The script evaluate.py runs evaluations for all models. To make dropping checkpoints of models to be evaluated easier, I have included an Evaluation config in the configs folder. Here, you can very easily place the checkpoints of the models (blockgpt, diffcast+phydnet or diffcast+blockgpt) and run evaluate.py. I have only run evaluations previously on quechua. If you wish to run the evaluation job script, please change the job config to request for more cpu memory. 


