```markdown
Dear all,

Homework 2 has been posted.

In this homework, you will **replicate** the DeepLOB model from Zhang et al. (2019), and you may also **refine** the model optionally if you would like to explore further. The goal is to reproduce the main experimental setting discussed in class and evaluate the predictive performance of the model on limit order book data.

For the homework, please use the **FI-2010 dataset**, which is the benchmark high-frequency limit order book dataset used in the DeepLOB paper. We will follow **Setup 2** from the paper. According to the course slides, you should use:

- **Training data**: the first 7 days  
  Train_Dst_NoAuction_ZScore_CF_7.txt

- **Test data**: the last 3 days  
  Test_Dst_NoAuction_ZScore_CF_7.txt  
  Test_Dst_NoAuction_ZScore_CF_8.txt  
  Test_Dst_NoAuction_ZScore_CF_9.txt  

Dataset download link:  
https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649

Reference GitHub implementation:  
[https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Nets-for-Limit-Order-Books)

Please also pay close attention to the data format. In the raw FI-2010 files, each column represents one event, while rows represent features and labels. The slides note that rows 1–144 contain features, and the DeepLOB model uses the top 10 levels of ask price, ask volume, bid price, and bid volume as inputs. In many public implementations, you will need to transpose the raw dataset so that rows represent events and columns represent features.

The input to the model follows the lecture discussion of DeepLOB: each sample is built from the 100 most recent LOB states, and each state contains 40 features from the top 10 bid/ask levels.

Please carefully read Lecture 4, pages 75–81, since these pages contain the dataset description, feature/label structure, label definition, and the homework requirements.

Bonus tasks:
- test different prediction horizons by trying different values of \( kk \) and analyze the pattern of predictability;
- explore asset-specific vs. universal modeling.

For submission, please include a `.ipynb` notebook, similar to what you submitted for Homework 1. Your notebook should clearly describe the data files used, preprocessing steps, model settings, evaluation metrics, and a comparison between your results and those reported in the paper.

Best,
```