# perovsktes-SiteHDNN

#Code for Data Preprocessing
1.We hereby declare that RuNNerUC.py file is from Professor Jorg Behler ,not original by the author. RuNNerUC.py can convert structural files into the files, which could be recognized by RuNNer software.
2.1-vasp2SiteGTSD_formation.sh file is used to convert vasp structure files into Site-GTSD files , accompanied by formation energy labels, for Site-HDNN model training and application.
3.2-vasp2SiteGTSD_formation.sh file is used to convert vasp structure files into Site-GTSD files , accompanied by bandgap labels, for Site-HDNN model training and application.
4.3-vasp2GTSD_formation.sh file is used to convert vasp structure files into GTSD files , accompanied by formation energy labels, for HDNN model training and application.
5.The 1-vasp2SiteGTSD_formation.sh, 2-vasp2SiteGTSD_formation.sh, 3-vasp2GTSD_formation.sh files need to be used together with RuNNerUC.py file.
6.The examples used by programs 1-vasp2SiteGTSD_formation.sh, 2-vasp2SiteGTSD_formation.sh, 3-vasp2GTSD_formation.sh are in file example-vasp2runner.rar
7.The model-evaluation.py file is used to evaluate the performance of the model.

#DataSet
8.The Atomly.rar file contains halide perovskite data from the Atomly database, which is used to evaluate the generalization capability of the Site-HDNN model.
9.The Dynamic simulation.rar contains halide perovskite data from the simulation of Quantum Dynamics, including a-MAPbI3, b-MAPbI3, a-FAPbI3, b-FAPbI3, a-CsPbI3, r-CsPbI3, which is used to train the Site-HDNN model.






