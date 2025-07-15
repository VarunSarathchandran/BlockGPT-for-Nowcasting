BlockGPT: Readme to follow


Instructions to create the environment

Step 1: conda env create -f blockgpt_env.yml
Step 2 (only if you get the error importing cached_download in dynamic_utils.py when you run training):
        Replace this line in dynamic_utils.py (path to the file will be mentioned in the error log)
        from huggingface_hub import cached_download, hf_hub_download, model_info
        with 
        from huggingface_hub import hf_hub_download, model_info
Step 3: pip install imageio==2.33.0 --no-deps 
Step 4: pip install ema-pytorch

