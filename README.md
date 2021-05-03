# Sim-to-Real semantic segmentation for humanoid soccer robots
This project submitted at 24th RoboCup International Symposium as an open-source project.

| Caption  | Download Link |
| ------------- | ------------- |
| Synthetic Dataset  | [GDrive](https://drive.google.com/file/d/1XinZHuwVtGzm4pzWZvbqeROO_O4IGGim/view?usp=sharing)  |
| Real Dataset  | [GDrive](https://drive.google.com/file/d/1v0EUAB8FtaIa078qA0CB7h3fnlGPJhys/view?usp=sharing)  |
| Simulator  | [Simulator](https://drive.google.com/file/d/1mKEscPXa2ESHYhZAZ5DIR49AQJiKfwrN/view?usp=sharing)  |
| See Video  | [Youtube](https://youtu.be/y-lblnwJhKM)  |


## Abstract
As the humanoid soccer robots are highly dependent on their visual perception to recognize different elements in the competition environment, they must be provided with effective algorithms running on their low-end hardware. This project proposes employing U-Net architecture to perform semantic segmentation as a solution to meet the robot's needs during the match. However, there are challenges especially in terms of a proper and subject-related dataset. Besides annotating the dataset manually for semantic segmentation has some considerable difficulties. To give an instance, generating masks manually is time-consuming and error-prone. Thus, we took advantage of realistically simulating the match environment and generating large amounts of data and their corresponding mask automatically in a short amount of time. The synthetic dataset was used to train the model, and to evaluate the model, a manually annotated dataset, consisting of real images captured from humanoid robots in the match environment, was produced. Although the proposed Sim-to-Real approach generated a dataset close to the real one, the results were not satisfying. To deal with this problem, the transfer learning technique was used to fine-tune the network and achieve better accuracy in real game conditions


## Dataset
The synthetic dataset is composed of 10,000 images along with their true masks (pixel-wise labels). Also, The real dataset has consisted of 250 real images which manually segmented.

Download Synthetic Dataset [GDrive](https://drive.google.com/file/d/1XinZHuwVtGzm4pzWZvbqeROO_O4IGGim/view?usp=sharing)

Download Real Dataset [GDrive](https://drive.google.com/file/d/1v0EUAB8FtaIa078qA0CB7h3fnlGPJhys/view?usp=sharing)


## Simulation
In this paper, to simulate a more realistic environment in the humanoid soccer league a number of artificially created changes have been put in place and numerous advances have been achieved. All the objects (i.e., robots, balls) and textures on the floor and wall spawn randomly during generating the dataset. Besides, a camera was attached to one of the robots, so all the images were taken in different view angles and locations on the field of play. Applying all these conditions in the simulator, 10,000 images with the shape of (480,640) pixels were generated which were annotated automatically. All the rendered images were converted to a mask, based on the blueprints visual scripting of the UE, which provides text files including id numbers between 0 to 5. Each id number specifies a class; {0: Ball, 1: Field, 2: Robots, 3: Lines, 4: Background, 5: Goals}.

Download Simulator to generatee synthetic dataset [Simulator](https://drive.google.com/file/d/1mKEscPXa2ESHYhZAZ5DIR49AQJiKfwrN/view?usp=sharing)


## Video
The proposed model was tested on a humanoid soccer robot with an Intel® Core™ i3-6100U Processor (3M Cache, 2.30 GHz), and turned out to be competent enough to process the unseen real data with a sample rate of 7 fps. 

The network outputs video in the real environment is uploaded in the [Youtube](https://youtu.be/y-lblnwJhKM)
