

# Project Setup and Usage

## Environment Setup
To set up the environment, first install the dependencies from `requirements.txt`:

```
pip install -r requirements.txt
````

In addition, please follow the installation instructions from the following repositories, which this code builds on top of:

* [icl-coverage](https://github.com/Shivanshu-Gupta/icl-coverage)
* [in-context-learning](https://github.com/Shivanshu-Gupta/in-context-learning)
* [DataInf](https://github.com/ykwon0407/DataInf)

---

## Running Experiments

### Mixture of Tasks (MoT)

Run the following file to reproduce MoT experiments:

```
sh MoT/icl-coverage/src/indirect.sh
```

### Noisy ICL

Run the following file to reproduce Noisy ICL experiments:

```
sh Noisy_ICL/icl-coverage/src/run_all.sh
```


