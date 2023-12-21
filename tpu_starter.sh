for var in "$@"
do
    until gcloud compute tpus tpu-vm create $var --zone=europe-west4-a --accelerator-type=v3-8 --version=tpu-vm-base; do sleep 5; done
done