# Malaria Detection Using Deep Learning

**MIT Capstone Project | Computer Vision | Convolutional Neural Networks | Healthcare AI**

## Project Overview

This project investigates the application of deep learning to malaria parasite detection in microscopic images of red blood cells.

Manual microscopic examination of blood smears is labour-intensive and requires experienced laboratory personnel. Deep learning offers a potential approach to supporting image screening, improving consistency and reducing the burden of manual image assessment.

The project develops and evaluates six deep learning model configurations, comparing custom Convolutional Neural Networks (CNNs), data augmentation, transfer learning and fine-tuning.

The best-performing model achieved **98.85% test accuracy** and **99.08% recall for parasitized cells**.

## Project Objectives

- Develop a binary image classification model to distinguish parasitized from uninfected red blood cells.
- Investigate the impact of architectural modifications, Batch Normalization, LeakyReLU and data augmentation.
- Evaluate transfer learning and selective fine-tuning using pretrained VGG16.
- Compare model performance using validation loss, test accuracy, precision, recall, F1-score and confusion matrices.
- Identify a suitable candidate for further development as a malaria cell-image screening aid.

## Dataset

The dataset comprises **27,558 colour microscopy images** of individual red blood cells, divided into two classes:

- **Parasitized:** Red blood cells infected with Plasmodium parasites.
- **Uninfected:** Red blood cells not infected with Plasmodium parasites.

| Dataset | Parasitized | Uninfected | Total |
|:--------|------------:|-----------:|------:|
| Original training set | 12,582 | 12,376 | 24,958 |
| Test set | 1,300 | 1,300 | 2,600 |
| **Total** | **13,882** | **13,676** | **27,558** |

The original training dataset was further divided into training and validation subsets using a stratified 80:20 split:

- Training: 19,966 images
- Validation: 4,992 images
- Testing: 2,600 images

## Methodology

The project follows an end-to-end deep learning workflow:

1. Dataset verification and exploratory data analysis.
2. Image preprocessing, resizing and pixel normalisation.
3. Stratified training and validation split.
4. Baseline CNN development.
5. Architectural optimisation and regularisation experiments.
6. Image data augmentation.
7. VGG16 transfer learning and selective fine-tuning.
8. Model evaluation, comparison and selection.
9. Conclusions and practical implementation recommendations.

Images were resized to **64 × 64 × 3** for the custom CNN models. Early stopping was used during training to monitor validation performance and restore the best-performing model weights.

## Models Evaluated

Six deep learning model configurations were developed and compared:

| Model | Approach |
|:------|:---------|
| Base CNN | Baseline convolutional neural network |
| Model 1 | Expanded CNN architecture |
| Model 2 | CNN with Batch Normalization and LeakyReLU |
| Model 3 | CNN with data augmentation |
| VGG16 – Transfer Learning | Pretrained frozen convolutional base |
| VGG16 – Fine-Tuning | Selective fine-tuning of the final convolutional block |

## Model Performance

| Model | Min. Validation Loss | Test Accuracy | Test Loss | False Positives | False Negatives |
|:------|---------------------:|--------------:|----------:|----------------:|----------------:|
| Base CNN | 0.0741 | 97.31% | 0.0681 | 47 | 23 |
| Model 1 – Expanded CNN | 0.0536 | 98.31% | 0.0486 | 32 | 12 |
| Model 2 – BatchNorm + LeakyReLU | 0.0547 | 98.65% | 0.0423 | 18 | 17 |
| **Model 3 – Data Augmentation** | **0.0498** | **98.85%** | **0.0397** | **18** | **12** |
| VGG16 – Transfer Learning | 0.1271 | 93.96% | 0.1585 | 91 | 66 |
| VGG16 – Fine-Tuning | 0.0841 | 97.00% | 0.1066 | 44 | 34 |

## Final Model Selection

**Model 3 – Data Augmentation** was selected as the preferred candidate for further evaluation, primarily because it achieved the lowest minimum validation loss.

### Key Performance Metrics

- **Test accuracy:** 98.85%
- **Test loss:** 0.0397
- **Minimum validation loss:** 0.0498
- **Parasitized-class recall:** 99.08%
- **Correctly classified images:** 2,570 out of 2,600
- **False positives:** 18
- **False negatives:** 12

### Confusion Matrix

| Actual / Predicted | Uninfected | Parasitized |
|:-------------------|-----------:|------------:|
| Uninfected | 1,282 | 18 |
| Parasitized | 12 | 1,288 |

Model 3 matched Model 1 for the lowest number of false negatives while producing fewer false positives.

These results demonstrate the importance of considering class-specific classification errors alongside overall accuracy, particularly in screening-oriented healthcare applications.

## Key Findings

- Custom CNN architectures achieved strong classification performance on the available dataset.
- Data augmentation was associated with improved validation and test performance under the experimental conditions evaluated.
- Frozen VGG16 underperformed the custom CNN architectures.
- Selective fine-tuning improved VGG16 test accuracy from **93.96% to 97.00%**, an increase of **3.04 percentage points**.
- Model selection should consider validation performance and the consequences of classification errors rather than relying on accuracy alone.

## Practical Implications and Limitations

The results demonstrate the potential of deep learning for malaria cell-image classification and provide a foundation for investigating AI-assisted microscopy screening.

However, the model classifies individual red blood cell images and does not establish a patient-level malaria diagnosis.

Furthermore, the test dataset was used to evaluate multiple candidate models. The reported results therefore represent comparative experimental findings rather than a final independent assessment of the selected model.

Further development would require:

- Independent external validation across different laboratory and imaging conditions.
- Sensitivity-focused evaluation and classification threshold optimisation.
- Patient-level dataset separation where patient identifiers are available.
- Assessment of computational requirements, laboratory workflows and implementation costs.
- Further investigation of transfer learning and alternative pretrained architectures.

**This project is a research and educational prototype, not a clinically validated diagnostic tool.**

## Technologies and Tools

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Google Colab

## Repository Contents

| File | Description |
|:-----|:------------|
| [Malaria_Detection_CNN.ipynb](Malaria_Detection_CNN.ipynb) | Complete project notebook, including exploratory data analysis, preprocessing, model development, training, evaluation, visualisations and conclusions |
| README.md | Project overview, methodology and key findings |

## Running the Notebook

The notebook was developed in Google Colab and includes dataset preparation, model training and evaluation.

To reproduce the experiments:

1. Open `Malaria_Detection_CNN.ipynb` in Google Colab.
2. Enable a GPU runtime where available.
3. Obtain the malaria microscopy image dataset separately.
4. Prepare and upload the `cell_images.zip` dataset with the expected training and testing directory structure.
5. Execute the notebook cells sequentially.

The notebook contains saved experimental outputs, allowing readers to review the results without retraining the models.

## Conclusion

This project demonstrates an end-to-end application of deep learning to a healthcare-related computer vision problem, combining exploratory data analysis, CNN development, data augmentation, transfer learning and comparative model evaluation.

The findings highlight the potential of custom CNN architectures for specialised microscopy image classification and the importance of responsible, validation-driven model selection in healthcare AI.

## Author

**Joanna Cookney**

MIT Capstone Project – Deep Learning and Computer Vision
