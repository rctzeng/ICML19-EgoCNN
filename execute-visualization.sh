####### Train the network #######
# Proprocess to generate the compound datasets
python preprocess-dataset.py -n Compound_Asym-Sym -k 4
python preprocess-dataset.py -n Compound_Alk-Alc -k 4
# You can use `-m` to speificy the network architecture, default=6L is already yielding nice results
python run-ego-cnn.py -n Compound_Asym-Sym -k 4 -p 4 -g 1 -f 0.99 -b 10 -l 1
python run-ego-cnn.py -n Compound_Alk-Alc -k 4 -p 4 -g 1 -f 0.99 -b 10 -l 1

####### Visualization Option 1: Attention Layer + Transposed Convolution #######
# requires to add Attention Layer and retrain the network
python run-visualization-model.py -n Compound_Asym-Sym -k 4 -p 4 -g 1 -f 0.99 -b 10 -l 1
python run-visualization-model.py -n Compound_Asym-Sym -k 4 -p 4 -g 1 -f 0.99 -b 10 -l 1
python run-visualization-model.py -n REDDIT-BINARY -k 4 -p 4 -g 1 -f 0.99 -b 10 -l 1
# You can use `-t` to specify the threshold = [0.6,0.7,0.8], default=0.8 is already yielding nice results
python plot-critical-structure-Attention.py -n Compound_Asym-Sym -k 4 -p 4 -g 1 -f 0.99 -L 6
python plot-critical-structure-Attention.py -n Compound_Alk-Alc -k 4 -p 4 -g 1 -f 0.99 -L 6
python plot-critical-structure-Attention.py -n REDDIT-BINARY -k 4 -p 4 -g 1 -f 0.99 -L 6

# Option 2: Grad-CAM(ICCV'17) + Transposed Convolution
python plot-critical-structure-GradCAM.py -n Compound_Asym-Sym -k 4 -p 4 -g 1 -f 0.99 -L 6
python plot-critical-structure-GradCAM.py -n Compound_Alk-Alc -k 4 -p 4 -g 1 -f 0.99 -L 6
python plot-critical-structure-GradCAM.py -n REDDIT-BINARY -k 4 -p 4 -g 1 -f 0.99 -L 6