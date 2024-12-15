**Description**


This model was trained based on LES data obtained from VFS(Virtual Flow Simulator Open-source code) and is utilized for flow field prediction in wind turbines in a significantly shorter amount of time compared to numerical simulations (LES).

You can find the required libraries in the requirements.txt file. It may be necessary to modify the turbine configuration in the [floris library](https://nrel.github.io/floris/) to adjust the library's functionality to that of a special turbine.
Turbine type, each turbine's location (layout_x, layout_y), and the flow field properties are included in **gch.yaml** file. This file is fed to the input_file.yaml as the floris input.


Files such as **seimens.yaml** and **vestas.yaml** are turbine brand-specific and include information such as hub height, rotor diameter, power(based on wind speed), and thrust coefficient. 


The **param.yaml** files are case-specific auxiliary data injected into the 2D Unet model (MLP) during training. These files include x, y, yaw angle, rotor diameter, and wind speed.


To initiate the training process, it's essential to define the Turbine type (using the brand-specific data) in the gch.yaml file, declare domain size and number of computational nodes in **input_file.yaml**, and generate the floris data with the corresponding configurations using **fldataset.py**. 
The **unet.py** file can be used to train the model from scratch.


** Reference **

C. Santoni, D. Zhang, Z. Zhang, D. Samaras, F. Sotiropoulos, A. Khosronejad; Toward ultra-efficient high-fidelity predictions of wind turbine wakes: Augmenting the accuracy of engineering models with machine learning. Physics of Fluids 1 June 2024; 36 (6): 065159. https://doi.org/10.1063/5.0213321
