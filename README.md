# AI4AMP_predictor
### AI4AMP is a sequence-based antimicrobial peptides (AMP) predictor based on PC6 protein encoding method [[link]](https://github.com/LinTzuTang/PC6-protein-encoding-method) and deep learning.
##### AI4AMP (web-server) is freely accessible at http://symbiosis.iis.sinica.edu.tw/PC_6/

### Here we give a quick demo and command usage of our AI4AMP model.  
### 1. quick demo of our PC6 model
##### For quick demo our model, run the command below
```bash 
bash AI4AMP_predictor/test/example.sh
```
##### The input of this demo is 10 peptides (```test/example.fasta```) in [FASTA](https://en.wikipedia.org/wiki/FASTA_format) format.
##### The prediction result (```test/example_output.csv```) below shows prediction scores  and whether the peptide is an AMP in table.
![](https://i.imgur.com/xLjlGHV.png)
### 2. command usage
##### Please make sure your working directory access to  ```PC6/PC6_predictor.py``` and execute command like the example below
```bash
python3 PC6_predictor.py -f [input.fasta] -o [output.csv]
```
##### -f : input peptide data in FASTA format
##### -o : output prediction result in CSV 
##
### AI4AMP deep neural network model architecture
##### The model architecture consists of one convolution layer, one long short-term memory (LSTM) layer, and one dense layer, which is a typical architecture in natural language processing (NLP) tasks.
![](https://i.imgur.com/HXciubr.png)
##### The figure above shows AI4AMP model architecture. After PC6 encoding, protein sequences will pass through one convolution layer, one LSTM layer, and one dense layer.

