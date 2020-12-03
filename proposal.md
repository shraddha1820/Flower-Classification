# A Visual Vocabulary for Flower Classification

Digital Image Processing(CSE/ECE 478)

Based on http://www.robots.ox.ac.uk/~men/papers/nilsback_cvpr06.pdf
<br/>
<br/>
## Team members:

Aditya Mohan Gupta - (2019201047)
\
Aditya Gupta - (2019201067)
\
Divy Kala - (2019201022)
\
Shraddha - (2019201001)


## 1. Goal of the project:

- The project aims at developing a flower classification model capable of classifying flowers with significant visual similarity on a challenging dataset. 
- The dataset comprises flowers indistinguishable from color alone, and therefore we employ a model which takes cues from various aspects of a flower such as shape, scale, and texture. 
- The goal of our project is to develop a flower classification model superior to baseline models classifying flowers on color cues alone.


## 2. Problem definition:

- Flower classification classifies images of flowers into their flower categories.
- The problem is particularly challenging owing to the small inter-class variation and large intra-class variation. Subtle differences are important in determining the correct flower category. For instance, violets can be both white and violet in color, and therefore exhibit a large intra-class variance. This observation calls for a multimodal classification method for classifying flowers.


## 3. Results of the project:

- The expectation includes to develop a flower classification model superior to baseline models classifying flowers on color cues alone to have better performance metrics.
- Results of the project defines the accuracy of the classifier based on some performance metrics.Performance metrics that can be used are
  - Precision.
  - Recall.
  - F1 score.
- The expectation is to get good accuracy by varying the hyperparameters and features used. 

## 4. What are the project milestones and expected timeline?
  <pre>
  - Understanding the data                - 22 Oct 2020
  - Preprocessing and feature extraction  - 30 Oct 2020
  - Fitting training models               - 12 Nov 2020
  - Analysing the results                 - 16 Nov 2020
  - Final Report                          - 18 Nov 2020
  </pre>


## 5. Is there a dataset that you require? How do you plan to get it?

- Link to dataset: [http://www.robots.ox.ac.uk/~vgg/data/flowers/17/](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
- Dataset Size : 1362 images
- consists of 17 species of flowers with 80 images of each
- The large intra-class variability and the sometimes small inter-class variability makes this dataset very challenging (some classes cannot be distinguished on colour alone (e.g. dandelion and buttercup), others cannot be distinguished on shape alone (e.g. daffodils and windflower).
