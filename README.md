# Shelly 3EM downloader and plotter

## Downloading data

To download data and store it on an SFTP server, run the following command:

```shell
docker run --rm -it \
  -v ~/.ssh/id_rsa:/mnt/id_rsa \
  ghcr.io/soerface/power-plot \
  sftp://username@host:22/home/path/ \
  --host 192.168.178.99 \
  --download-only \
  --ssh-key-path /mnt/id_rsa 
```

It's recommended to do this daily to get the best resolution of your data
(1 minute, after one or two days the resolution will be reduced to 10 minutes).

## Plotting data

This example plots data from a directory that needs to be mounted into the container.
Not all options here are required; please check --help for more information.

```shell
docker run --rm -it \
  -v ~/data/input:/mnt/input \
  -v ~/data/output:/mnt/output \
  ghcr.io/soerface/power-plot \
  /mnt/input/ \
  --dark-theme \
  --output /mnt/output/plot.svg \
  --from "2024-05-08" \
  --to "2024-05-10 12:00" \
  --sample-rate "1h" \
  --plot-phases
```