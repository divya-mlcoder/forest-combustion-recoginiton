# forest-combustion-recoginiton

# INTRODUCTION


Emergency situations like floods, earthquakes and fires pose a big threat to public health and safety, property and environment. Fire related disasters are the most common type of Emergency situation which requires thorough analysis of the situation required for a quick and precise response. The first step involved in this process is to detect fire in the environment as quickly and accurately as possible.

Fire Detection in most places employs equipment like temperature detectors, smoke detectors, thermal cameras etc. which is expensive and not available to all. But, after the advent of advanced image processing and computer vision techniques, detection of fire may not require any equipment other than cameras. Due to this expeditious development in vision-based fire detection models, there is a particular inclination towards replacing the traditional fire detection tools with vision-based models. These models have many advantages over their hardware based counterparts like accuracy, more detailed view of the situation, less prone to errors, robustness towards the environment, considerably lower cost and the ability to work on existing camera surveillance systems.

Therefore, our approach is to employ state-of-the-art CNNs to distinguish between images that containing fire and images that do not and build an accurate fire detection system. To make these models more robust, we use a custom-made image dataset containing images with numerous scenarios.

# Overview

Nowadays,Forest fires predic on combines weather factor, rain, dryness of flammable items, types of flammable items, and igni on sources to analyze and predict the combus on risks of flammable items in the forest. Forest fire predic on has developed rapidly in various countries in the world since its incep on in the 1920s. Taiwan's forestry department currently uses the study results of Hsiao (2003). Hsiao's study used a given day's highest temperature, temperature varia on, accumulated period without rainfall, and drought index as weather factors to derive forest fire incident in a logis cal regression model, and built a forest fire recogni on probability model and covolua on neural netwrok. Hsiao also considered space and  me varia ons in weather factors, used GIS systems to conduct temperature and rainfall space- me es mates, and es mated forest fire hazard ra ng predic ons for forest in Taiwan on a given day.

# Purpose

Our aim from the project is to make use of pandas, numpy libraries from python to extract the libraries for machine learning for the forest fire reconiza on. Secondly, to learn how to build model using ImageDataGenerater hypertune the forest with and without fire images using Convolu on neural netwrok(CNN) machine learning algorithm. And in the end, to predict whether the image is with fire or not using video analysis technique of combining the predic ons from machine learning algorithms and giving the alert message to the email address.

# LITERATURE SURVEY

Data mining is the process of analyzing data from diﬀerent perspec ves and extrac ng useful knowledge from it. It is the core of knowledge discovery process. The various steps involved in extrac ng knowledge from raw data as depicted in figures . Diﬀerent data mining techniques include classifica on, clustering, associa on rule mining, predic on and sequen al paerns, neural networks, regression,CNN,RNN etc. Classifica on is the most commonly applied data mining technique which employs a set of preclassified examples to develop a model that can classify the popula on of records

at large.This approach frequently employs Decision tree based classifica on Algorithm. In classifica on, a training set is used to build the model as the classifier which can classify the data images/items into its appropriate classes.A test set is used to validate the model.

# Proposed Solution

Machine Learning (Convolutional neural network architectures)

According to the principle of object detec on algorithms, the flow of image fire detec on algorithms based on convolu onal neural networks is designed in the detec on CNN has func ons of region proposals, feature extrac on and classifica on. Firstly, The CNN takes an image as input and outputs region proposals by convolu on, pooling, etc. Secondly, the region-based object detec on CNN decides the presence or absence of fire in proposal regions through convolu onal layers, pooling layers, fully-connected layers, etc.

Coming to project build first create a dataset with train and test folders then Data pre-processing step (the externally important step in any project). I highly recommend going through the "Basics of image processing using Python we use Keras" ImageDataGenerator class to perform data augmenta on. i.e, we are using some kind of parameters to process our collected data. The word “augment” means to make something “greater” or “increase” something (in this case, data). The ImageDataGenerator transforms each image in the batch by a series of random transla ons, these transla ons are based on the arguments and then Apply ImageDataGenerator Func onality to trainset and testset. Aer that Build a model: in that Impor ng the libraries, Ini alizing the model, Adding CNN (Convolu on Neural Network) Layers, Adding Dense layers. Aer building a model next step is video processing for the video processing used the tool openCV. this openCV is useful to genarate a video and predict the image. aer predic on will send an email with an alert message .if we found any place with fire inside the forest we send email with an alert message.

# THEORETICAL ANALYSIS

Forest fire destroys millions of hectares of forest, pollutes the environment, causes severe casual es, and has a significant economic impact on government budgets every year. Detec ng a wildfire promptly, before it is out of control, is s ll a diﬃcult challenge. According to the combus on materials, a wildfire is generally classified into three types: underground fire, surface fire, or crown fire. An underground fire is caused by spontaneous combus on or combus on in other channels aer coal strata meet combus on condi ons under the surface; and a surface fire, when not acted upon by external forces, usually spreads along the surface of forest areas. The surface fire is easily aﬀected by wind, causing the flames to disperse everywhere, eventually falling on the crown and branches, causing a crown fire to occur. Crown fires have excep onally high temperatures and ferocious behavior, which makes them challenging to be ex nguished, making them extremely dangerous. In addi on, a crown fire always spreads more than 100  mes faster than a surface fire and is more destruc ve. Thus, because of this phenomenon, it is essen al to determine the type of wildfire as early as possible in early wildfire detec on, for the sake of adop ng eﬃcient strategies to fight the wildfire and reduce the casual es and economic losses.

This sec on describes the projected system for eﬃcient forest fire detec on. The input fed to the system is the spa al data matching to the forest regions. Data mining, image processing and ar ficial intelligence techniques support the detec on of fire from the images of spa al data.

Convolutional neural networks. Sounds like a weird combination of biology and math with a little CS sprinkled in, but these networks have been some of the most influential innovations in the field of computer vision and image processing. The Convolutional neural networks are regularized versions of multilayer perceptron (MLP). They were developed based on the working of the neurons of the fire visual cortex.

Let’s say we have a color image in JPG form and its size is 480 x 480. The representative array will be 480 x 480 x 3. Each of these numbers is given a value from 0 to 255 which describes the pixel intensity at that point. RGB intensity values of the image are visualized by the computer for processing.

# The objective of using the CNN:

The idea is that you give the computer this array of numbers and it will output numbers that describe the probability of the image being a certain class (.80 for a forest fire, .15 for a forest without fire, .05 for a forest fire etc.). It works similar to how our brain works. When we look at a picture of a forest, we can classify it as such if the picture has identifiable features such as with fire or 4 with out fire. In a similar way, the computer is able to perform image classification by looking for low-level features such as edges and curves and then building up to more abstract concepts through a series of convolutional layers. The computer uses low-level features obtained at the initial levels to generate high-level features such as forest with fire or without fire to identify the object. The image is resized to an optimal size and is fed as input to the convolutional layer. Let us consider the input as 32x32x3 array of pixel values.There exists a filter or neuron or kernel which lays over some of the pixels of the input image depending on the dimensions of the Kernel size. Let the dimensions of the kernel of the filter be 5x5x3.

# EXPERIMENTAL INVESTIGATION

In this project we created a dataset with collec on of forest images with fire and without fire. In the forest dataset we created two sub folders one is testset another one traine set. In each folder we created another two sub folders one is forest with fire and forest without fire.

# RESULT

To measure the performance of the proposed model, the images were collected from Internet. The figures shows the variety of forest fire condi on images used in the test. A true-posi ve was counted if an image frame had fire pixels, and it was determined by the proposed model to be fire. In contrast, false-posi ve was counted, if the image frame has no fire, and the result was determined as a fire. finally we got alert message.

The above shows the prototype and the conversion models. The modules work independently of each other. Each modules converts video into frames and each frame is converted to bitmap image. In above images(Theore cal Analysis) the three clones of the bitmap is taken and converted into Y, Cb, Cr channels.

# ADVANTAGES

●The CNN can consider the correlation of adjacent spatial information, it has advantages in the study of problems with spa al and geographical correlation characteristics. Second, the CNN preserves the spatial relationships between pixels by learning the internal feature representations from factor vectors. The process of DL reveals the deep features and can distinguish the diﬀerences between diﬀerent geographical units. The CNN was used to conduct multiple convolution and pooling operations to extract the characteristics. As the convolutions and pooling increased, these features became more advanced and more abstract. These abstract features depicted the degree of forest fire susceptibility, which was the decisive factor for determining forest fire susceptibility.

●Areas destroyed by these fires are large and produce more carbon monoxide than the overall automobile traﬃc.

●Monitoring of the potential risk areas and an early detection of fire can significantly shorten the reaction  me and also reduce the potential damage as well as the cost of fire fighting.

●less expensive.

●low consumption.

●Reduce death rate.

●Simple, fast and easy to implement.

●Reduces significant economic impact on government budgets every year.

●Reduces the pollution of the environment.

# DISADVANTAGES

●Possibility of lack of appropriate animals for special forests.

●Determining climate conditions, daily temperature diﬀerences, Seasonal normal temperature values etc.. are problematic.

●User can make mistakes while typing a message format.

●Moreover, each needs to be changed periodically,but capturing the MBS to do this is not easy.

●Network issues.

# APPLICATIONS

●If a node detects a fire it sends an alert message to the email.

●Need to detect fire before it spreading.

●It is used for Health, Military, Commercial applications.

●It use Multi -hop communication for large forest

●And also used for Environmental applications like Habitat Monitoring

and Forest fire detec on.

# CONCLUSION

This project proposes an eﬀective forest fire detection method using CNN and max pooling opera on techniques including movement containing region detection based on color segmentation. The performance of the proposed algorithm is tested on data set consisting of forest with and without fire images from the internet. TP-rate and TN-rate were calculated. The results show that the proposed algorithm achieves good detection rates. These results indicate that the proposed method is accurate and can be used in automatic forest fire-alarm systems. For future work, the system could be improved by using a combination of rules of diﬀerent color spaces; however, the challenge is selecting the right rules from diﬀerent color spaces to build the method.

# FUTURE SCOPE

●In future, we will be updating the system with additional features like increase the range of sensing of the sensor, monitoring the count of animals present in the forest and can be prevented from being endangered.

●It can be further expanded with a voice interactive system facility.

●Right now provide the control of alert message for the one device but it can be designed for more number of devices.

●A feedback system also be included which provide state of a device (whether it is on/oﬀ) to the remote device.

●To overcome disadvantages and looking for implementation of automatic fire detection.

●In place of alerting to emails, mobile calls and messages we sensors.


# BIBLIOGRAPHY

●B. Ko and S. Kwak, “Survey of computer vision based natural disaster warning systems,” Op cal Engineering, vol. 51, no. 7, Ar cle ID070901,2012.
●J. R. Mar nez-de Dios, B. C. Arrue,A. Ollero, L. Merino,and F. G´omez-Rodr´ıguez, “Computer vision techniques for forest fire percep on,”Image and Vision Compu ng, vol.26,no.4,pp. 550–562,2008.

●Y.Meng, Y.Deng, and P.Shi,“Mapping Forest Wild fire Risk of the World,”in World At las of Natural Disaster Risk, P. Shi and R. Kasperson, Eds., pp. 261–275, Springer Berlin Heidelberg, Berlin,Germany,2015.

●P. M. Hanamaraddi, “A Literature Study on Image Processing for Forest Fire Detec on, ”IJITR,vol.4,pp.2695–2700,2016.

●Bui, D.T., K.T.T. Le, V.C. Nguyen, H.D. Le, and I. Revhaug. 2016. Tropical forest fire suscep bility mapping at the Cat Ba Na onal Park area, Hai Phong City, Vietnam, using GIS-based Kernel logis c regression. Remote Sensing.
●Z hang, X. 2007. Vegeta on map of the People’s Republic of China (1:1000000). Beijing: Geology Press (in Chinese).

●Ying, L., J. Han, Y. Du, and Z. Shen. 2018. Forest fire characteris cs in China: Spa al paerns and determinants with thresholds. Forest Ecology and Management 424

●Y. Wang and J. Ye, “Research on the algorithm of preven on forest fire disaster in the Poyang Lake Ecological Economic Zone, ”Advanced Materials Research, pp.5257–5260,2012.
