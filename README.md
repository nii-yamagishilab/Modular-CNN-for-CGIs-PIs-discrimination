# Modular-CNN-for-CGIs-PIs-discrimination

Implementation of the paper:  <a href="https://dl.acm.org/citation.cfm?id=3230863">Modular Convolutional Neural Network for Discriminating between Computer-Generated Images and Photographic Images</a> (ARES 2018).

You can clone this repository into your favorite directory:

    $ git clone https://github.com/nii-yamagishilab/Modular-CNN-for-CGIs-PIs-discrimination

## Requirement
- PyTorch 0.2
- TorchVision
- scikit-learn
- Numpy
- pickle

## Project organization
- Datasets folder, where you can place many datasets inside:

      ./datasets/<dataset1; dataset2..>
- Output folder, where the training outputs will be stored:

      ./output/<dataset1; dataset2..>
      
## Dataset
Each dataset has two parts:
- Photographic images: \<path-to-dataset\>/\<train;test;validation\>/PI
- Computer-generated images: \<path-to-dataset\>/\<train;test;validation\>/CGI

Moreover, splicing images will be stored in \<path-to-dataset\>/splicing

## Training
### + Train the feature extractor and classifier using Multilayer Perceptron (MLP)

### + Train the classifier using Linear Discriminant Analysis (LDA)

## Evaluating - Patch Level
### + Evaluating using MLP

### + Evaluating using LDA

## Evaluating - Full-Size Images
### + Evaluating using MLP

### + Evaluating using LDA

## Evaluating - Splicing Detection

#  Authors
- Huy H. Nguyen (https://scholar.google.com/citations?user=8q1km_cAAAAJ&hl=en)
- Ngoc-Dung T. Tieu
- Hoang-Quoc Nguyen-Son: (https://scholar.google.com/citations?user=UTODzwgAAAAJ&hl=en)
- Vincent Nozick (http://www-igm.univ-mlv.fr/~vnozick/?lang=fr)
- Junichi Yamagishi (https://researchmap.jp/read0205283/?lang=english)
- Isao Echizen (https://researchmap.jp/echizenisao/?lang=english)

# Reference
H. H. Nguyen, T. N.-D. Tieu, H.-Q. Nguyen-Son, V. Nozick, J. Yamagishi, and I. Echizen, “Modular Convolutional Neural Network for Discriminating between Computer-Generated Images and Photographic Images,” Proc. of the 13th International Conference on Availability, Reliability and Security (ARES 2018), 10 pages, (August 2018)

## TO BE CONTINUED
