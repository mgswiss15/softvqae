# README

## Environment

```
conda create --name compr
conda activate compr
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install wandb
pip install matplotlib
```

## Run

To run locally, edit the configuration file (e.g. `./configs/ae_run.ini`) or other and provide it at command line call.
Default configs are in `./configs/ae.ini` **you should not touch this**.
They are always read first and overwritten by whatever configs you provide in the `--config` file. 


```python train.py --cfile ./configs/ae_run.ini```

### Internal HPC set-up - extranals IGNORE this

To run a single job on baobab set up the configuration in the echos of `submit_job.sh` and execute it `./submit_job.sh`.
To launch multiple jobs set up the lists for the for loop variables in `submit_jobs.sh` and execute it `./submit_jobs.sh`


