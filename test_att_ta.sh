#!/bin/bash

export WANDB_API_KEY=afe1eb36cf1d27229ac6f273de34aee6d3955a73

python3 ta_attn_test.py
  
unset WANDB_API_KEY