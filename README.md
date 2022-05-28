# Attribution Visualization
Implementation for "Understanding contributing neurons via attribution visualization".  
AttrVis is an **optimization-based** attribution **visualization** method to understand the **implications** of contributing neuron features.  
The major advantage is the ability of producing noise-free visualizations, *i.e.,* using an optimizable mask to remove noise and artifacts in visualizations.  
<!-- ![Attribution visualization results](https://github.com/GlowingHorse/Attribution-Visualization/blob/main/data/attrVis.jpg) -->
<img src="https://github.com/GlowingHorse/Attribution-Visualization/blob/main/data/attrVis.jpg" width=50% height=50%>

AttrVis Quickstart
===
## Installation
0. It is safer to create a virtual environment before installing libraries in *requirements.txt*.
1. Install necessary libraries listed in *requirements.txt*.

## Usage
1. Run *gen_AS_baseline.py* to generate Aumann-Shapley attribution baseline.
2. Run *gen_attr_results.py* to calculate neuron attributions.
3. Run *attrVisGNet.py* to generate attribution visualizations.

## For mask perturbation
- The settings of `transforms` in *gen_attr_results.py* and fractal noise pyramid intensity (`image_sample`) in *./utils/render_baseline.py* can be finetuned according to network strucutre for better results.  
- The `lambda_param` in *./utils/render_baseline.py* is just the &lambda; in the manuscript. While the default setting is usually fine, for better visualization it needs to be adjusted based on your network and the data you are generating.  
- Although the current objective function `loss` in *./utils/render_baseline.py* can produce the best results, we still keep some other tested losses in comments, hoping to give you some new ideas. All these losses can be easily reused in similar visualization tasks.

## Others
- Some other attribution methods are provided in *gen_attr_results.py*, but generally, we find Aumann-Shapley method is a good to calculate neuron attributions.  
The attribution calculation framework is implemented with reference to [**DeepExplain**](https://github.com/marcoancona/DeepExplain).  
- When you try different networks that are provided in [**TensorFlow-Slim**](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models), the transform methods and some random image preconditioning settings should be changed accordingly.
- Other visualization ideas can also be found in [**lucid**](https://github.com/tensorflow/lucid). They have integrated many useful loss functions, regularizations, and preconditioning methods from a lot of literatures in CNN visualizations.
