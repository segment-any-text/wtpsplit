#!/bin/bash

TPU_VM_NAME="v3-8_4-1.13" # Name of the TPU VM
ZONE="europe-west4-a"     # Zone

# Create the TPU VM, retry if it fails
until gcloud compute tpus tpu-vm start "$TPU_VM_NAME" --zone="$ZONE"; do 
    sleep 1
done