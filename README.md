# Anti-Backdoor-Learning
A robust training methodology which helps the deep learning models to get trained robustly on poisoned data sets.

Recently, Deep Neural Networks(DNN) have been used in various sectors to solve many real-world problems. As these DNNs are gaining popularity, on the other hand, they are becoming vulnerable to security threats. Among the different security threats, backdoor attacks are emerging as a significant security threat to DNN. Although the existing defense methods, such as detection and erasing methods, give promising results by detecting and erasing backdoor triggers, there isa need for robust training methods. The following explanation gives an insight into why robust training methods are essential. Firstly, the detection models can only detect the presence of any backdoor trigger pattern in the training data/models. They cannot erase the backdoor trigger patterns. While wehave the erasing methods to erase the backdoor trigger pat-terns from the trained data/models, there is no guarantee that the erasing methods will erase all the backdoor trigger patterns our model learned. Secondly, the detection and erasing models are applied to the model before and after the training, respectively. Considering the above two points, this paper explores the concept of Anti-Backdoor Learning(ABL).
This methodology(ABL) helps the DNN models to unlearn the backdoor trigger patterns and learn the clean data, aiming to train the clean models. Using ABL, we can eliminate the
need for detecting and erasing methods as ABL is applied in the training phase. The goal of our paper is to try attacking the DNN models with different backdoor attacks. And try to use the ABL technique to overcome the simulated attack.

---

## Repoository Info:
* The folder `ABL-main` contains all the files related to implementing `ABL` methodolgy.
* The folder `CleanLabel` consists of all the files necessary to implement `Clean Label` attack.
* The folder `input-aware-backdoor-attack-release` consists all the files required to simulate the `Input Aware Dynamic Backdoor Attack`.

## Jupyter notebooks
* The file `AntiBackDoorLearning.ipynb` contains code related to applying ABL on various attack experimented such as `CleanLabel Attack`, `SIG Attack`, `BadNet Attack`.
* The file `CleanLabelAttack.ipynb` contains the required code to poison the dataset with `CleanLabel Attack` and how to train the model on the poisoned dataset.
* The file `Input_Aware_Dynamic_attack_iynb.ipynb` contains the execution of code under the folder `input-aware-backdoor-attack-release` in order compromise the model with `Input Aware Dynamic Backdoor Attack`.

---
### Contributors
* [Sai Praneeth Dulam](https://github.com/Saipraneeth99)
* [Bhargav Sai Gajula](https://github.com/bhargavsai2)
