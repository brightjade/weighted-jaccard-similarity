# Graphs are Documents: Predicting TCP Connection Attack via Weighted Jaccard Similarity

![weighted-jaccard-similarity](https://user-images.githubusercontent.com/33290571/153550664-22205ef0-bf0b-45d0-8cef-34dfb70def1f.png)

*This project was done while taking an AI607 course (Graph Mining) at KAIST.*

## Project Summary

Were there any malicious attacks? TCP connection attack prediction is a task of detecting attacks that may have occurred in the given connection histories. This is rather a difficult task because we do not know the number of attacks contained in each set of histories. While this may look like a graph problem, we convert these graph data to documents and compute similarities between each pair to identify patterns of a certain type of attack. For a similarity measure, we introduce a new, modified version of Jaccard similarity that is capable of handling skewed, imbalanced data. Our novel approach performs far better than random guesses, as well as vanilla Jaccard similarity, indicating that graphs can be processed as documents.
