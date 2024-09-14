First install and run the docker container with mmdeploy using:

- from the root directory of the repository run:
```bash
source runscripts/Deploy/01_docker_start.sh
```

Then setup the environment variables for the deployment using:

```bash
source root/workspace/mmdet/runscripts/Deploy/02_setup.sh
```

Finally deploy the model for Myriad using:

```bash
source /root/workspace/mmdet/runscripts/Deploy/03_deployer.sh
```