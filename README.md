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
**Note**: Parameters with detail explanation could be found in the corresponding source code.

First, train the the feature extractor and classifier using Multilayer Perceptron (MLP):

    $ python train.py --dataset ./datasets/dataset_1 --outf ./output --name dataset_1_output
    
After that, train the classifier using Linear Discriminant Analysis (LDA):

    $ python train_cls.py --dataset ./datasets/dataset_1 --outf ./output --name dataset_1_output --begin 0 --end 50

## Evaluating - Patch Level
**Note**: Parameters with detail explanation could be found in the corresponding source code.

Evaluating the network using MLP classifier:

    $ python test_patches_mlp.py --dataset ./datasets/dataset_1 --outf ./output --name dataset_1_output --id 50
    
Evaluating the network using LDA classifier:

    $ python test_patches_clf.py --dataset ./datasets/dataset_1 --outf ./output --name dataset_1_output --id 50


## Evaluating - Full-Size Images
**Note**: Parameters with detail explanation could be found in the corresponding source code.

Evaluating the network using MLP classifier:

    $ python test_full_mlp.py --dataset ./datasets/dataset_1 --outf ./output --name dataset_1_output --id 50
    
Evaluating the network using LDA classifier:

    $ python test_full_clf.py --dataset ./datasets/dataset_1 --outf ./output --name dataset_1_output --id 50
    
Random sampling can be activated by using this parameter:
    
    --random_sample <number of random samples>

## Evaluating - Splicing Detection
**Note**: Parameters with detail explanation could be found in the corresponding source code.

Evaluating the network using LDA classifier on splicing images can be done as follows:

    $ python splicing.py --dataset ./datasets/splicing --outf ./output --name dataset_1_output --stepSize 20 --id 50

## Authors
- Huy H. Nguyen (https://researchmap.jp/nhhuy/?lang=english)
- Ngoc-Dung T. Tieu
- Hoang-Quoc Nguyen-Son: (https://scholar.google.com/citations?user=UTODzwgAAAAJ&hl=en)
- Vincent Nozick (http://www-igm.univ-mlv.fr/~vnozick/?lang=fr)
- Junichi Yamagishi (https://researchmap.jp/read0205283/?lang=english)
- Isao Echizen (https://researchmap.jp/echizenisao/?lang=english)

## Reference
H. H. Nguyen, T. N.-D. Tieu, H.-Q. Nguyen-Son, V. Nozick, J. Yamagishi, and I. Echizen, “Modular Convolutional Neural Network for Discriminating between Computer-Generated Images and Photographic Images,” Proc. of the 13th International Conference on Availability, Reliability and Security (ARES 2018), 10 pages, (August 2018)

