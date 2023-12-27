for var in "$@"
do
    until gcloud compute tpus tpu-vm create $var --zone=europe-west4-a --accelerator-type=v3-8 --version=tpu-vm-pt-1.13; do sleep 3; done
done