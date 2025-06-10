# 0001: Run summary

Date: 2025-06-07

## Status
Summary

## Context
The first run has been completed and it seems that the loss has flatlined around epoch 40 before reasonable images have been generated. The images with different text prompts are different which suggests that the text embeddings are being considered by the model. 

## potential issues
- Learning rate too high or low
- more inference steps needed during training
- architectural issue
- Maybe 32x32 is just simply not enough pixels?
- maybe we need more base channells in the unet





