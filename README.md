# Zero-shot Text-to-SQL Learning with Auxiliary Task
Code for [Zero-shot Text-to-SQL Learning with Auxiliary Task] (https://arxiv.org/pdf/1908.11052.pdf)

## Usage

### Conda Environments
Please use Python 3.6 and Pytorch 1.3. Other Python dependency is in requirement.txt. Install Python dependency with:
```
	pip install -r requirements.txt
```

### Download Data
[Data](https://drive.google.com/file/d/1UQmL-F5tGUqAit35ybto7kk-3emkqtgE/view?usp=sharing) can be found from google drive. Please download them and extract them into root path.

### Generate our respilted WikiSQL data
```
	cd data_model/wikisql
	python make_zs.py
	python make_fs.py

```

### Run the model on original WikiSQL and our split

```
	cd TwoStream
	./run.sh  GPU_ID
```

## Acknowledgement
- This implementation is based on [coarse2fine](https://github.com/donglixp/coarse2fine).
- The preprocessing and evaluation code used for WikiSQL is from [salesforce/WikiSQL](https://github.com/salesforce/WikiSQL).