<p align="center">
<img src="./docs/img/nni_logo.png" alt="drawing" width="300"/>
</p>

-----------

[![MIT licensed](https://img.shields.io/badge/license-MIT-yellow.svg)](https://github.com/Microsoft/nni/blob/master/LICENSE)
[![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/Microsoft.nni)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=6)
[![Issues](https://img.shields.io/github/issues-raw/Microsoft/nni.svg)](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen)
[![Bugs](https://img.shields.io/github/issues/Microsoft/nni/bug.svg)](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen+label%3Abug)
[![Pull Requests](https://img.shields.io/github/issues-pr-raw/Microsoft/nni.svg)](https://github.com/Microsoft/nni/pulls?q=is%3Apr+is%3Aopen)
[![Version](https://img.shields.io/github/release/Microsoft/nni.svg)](https://github.com/Microsoft/nni/releases) [![Join the chat at https://gitter.im/Microsoft/nni](https://badges.gitter.im/Microsoft/nni.svg)](https://gitter.im/Microsoft/nni?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

NNI (Neural Network Intelligence) is a toolkit to help users run automated machine learning (AutoML) experiments. 
The tool dispatches and runs trial jobs generated by tuning algorithms to search the best neural architecture and/or hyper-parameters in different environments like local machine, remote servers and cloud.

<p align="center">
<img src="./docs/img/nni_arch_overview.png" alt="drawing"/>
</p>

## **Who should consider using NNI**
* Those who want to try different AutoML algorithms in their training code (model) at their local machine.
* Those who want to run AutoML trial jobs in different environments to speed up search (e.g. remote servers and cloud).
* Researchers and data scientists who want to implement their own AutoML algorithms and compare it with other algorithms.
* ML Platform owners who want to support AutoML in their platform.

## **Install & Verify**

**Install through pip** 	
* We support Linux and MacOS in current stage, Ubuntu 16.04 or higher, along with MacOS 10.14.1 are tested and supported. Simply run the following `pip install` in an environment that has `python >= 3.5`.	
```bash	
    python3 -m pip install --user --upgrade nni
```
Note: If you are in docker container (as root), please remove `--user` from the installation command.

**Install through source code**
* We support Linux (Ubuntu 16.04 or higher), MacOS (10.14.1) in our current stage. 
* Run the following commands in an environment that has `python >= 3.5`, `git` and `wget`.
```bash	
    git clone -b v0.4 https://github.com/Microsoft/nni.git
    cd nni	
    source install.sh	
```

**Verify install**	

The following example is an experiment built on TensorFlow. Make sure you have **TensorFlow installed** before running it.	
* Download the examples via clone the source code.	
```bash	
    git clone -b v0.4 https://github.com/Microsoft/nni.git
```
* Run the mnist example.
```bash
    nnictl create --config nni/examples/trials/mnist/config.yml
```

* Wait for the message `INFO: Successfully started experiment!` in the command line. This message indicates that your experiment has been successfully started. You can explore the experiment using the `Web UI url`.
```
INFO: Starting restful server...
INFO: Successfully started Restful server!
INFO: Setting local config...
INFO: Successfully set local config!
INFO: Starting experiment...
INFO: Successfully started experiment!
-----------------------------------------------------------------------
The experiment id is egchD4qy
The Web UI urls are: http://223.255.255.1:8080   http://127.0.0.1:8080
-----------------------------------------------------------------------

You can use these commands to get more information about the experiment
-----------------------------------------------------------------------
         commands                       description
1. nnictl experiment show        show the information of experiments
2. nnictl trial ls               list all of trial jobs
3. nnictl log stderr             show stderr log content
4. nnictl log stdout             show stdout log content
5. nnictl stop                   stop an experiment
6. nnictl trial kill             kill a trial job by id
7. nnictl --help                 get help information about nnictl
-----------------------------------------------------------------------
```

* Open the `Web UI url` in your browser, you can view detail information of the experiment and all the submitted trial jobs as shown below. [Here](docs/WebUI.md) are more Web UI pages.

<table style="border: none">
    <th><img src="./docs/img/webui_overview_page.png" alt="drawing" width="395"/></th>
    <th><img src="./docs/img/webui_trialdetail_page.png" alt="drawing" width="410"/></th>
</table>

## **Documentation**
* [NNI overview](docs/Overview.md)
* [Quick start](docs/GetStarted.md)

## **How to**
* [Install NNI](docs/Installation.md)
* [Use command line tool nnictl](docs/NNICTLDOC.md)
* [Use NNIBoard](docs/WebUI.md)
* [How to define search space](docs/SearchSpaceSpec.md)
* [How to define a trial](docs/howto_1_WriteTrial.md)
* [Config an experiment](docs/ExperimentConfig.md)
* [How to use annotation](docs/howto_1_WriteTrial.md#nni-python-annotation)
## **Tutorials**
* [Run an experiment on local (with multiple GPUs)?](docs/tutorial_1_CR_exp_local_api.md)
* [Run an experiment on multiple machines?](docs/tutorial_2_RemoteMachineMode.md)
* [Run an experiment on OpenPAI?](docs/PAIMode.md)
* [Run an experiment on Kubeflow?](docs/KubeflowMode.md)
* [Try different tuners and assessors](docs/tutorial_3_tryTunersAndAccessors.md)
* [Implement a customized tuner](docs/howto_2_CustomizedTuner.md)
* [Implement a customized assessor](examples/assessors/README.md)
* [Use Genetic Algorithm to find good model architectures for Reading Comprehension task](examples/trials/ga_squad/README.md)

## **Contribute**
This project welcomes contributions and suggestions, we use [GitHub issues](https://github.com/Microsoft/nni/issues) for tracking requests and bugs.

Issues with the **good first issue** label are simple and easy-to-start ones that we recommend new contributors to start with.

To set up environment for NNI development, refer to the instruction: [Set up NNI developer environment](docs/SetupNNIDeveloperEnvironment.md)

Before start coding, review and get familiar with the NNI Code Contribution Guideline: [Contributing](docs/CONTRIBUTING.md)

We are in construction of the instruction for [How to Debug](docs/HowToDebug.md), you are also welcome to contribute questions or suggestions on this area.

## **License** 
The entire codebase is under [MIT license](https://github.com/Microsoft/nni/blob/master/LICENSE)

