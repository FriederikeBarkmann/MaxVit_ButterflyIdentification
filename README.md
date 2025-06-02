# MaxVit_ButterflyIdentification

Scripts to train a MaxViT-T model on a butterfly dataset for species identification. 

A [MAxViT-T](https://arxiv.org/abs/2204.01697) model that was pre-trained on the ImageNet-1K dataset was finetuned using a dataset with over 500,000 images of 160 butterfly and moth species that occur in Austria. The dataset that was used is a subset of a dataset with over 182 species that was collected with the application ["Schmetterlinge Österreichs"](https://www.schmetterlingsapp.at/) of the foundation "Blühendes Österreich". It contains only those species with at least 50 images in the dataset. 10 % of the images were assigned as testdata with a random stratified appraoch prior to model training. 

The model was trained on the EuroHPC supercomputer LUMI, hosted by CSC (Finland) and the LUMI consortium. 

See this publication for more information on model training and the dataset: link to paper will be added when it is published. 

The dataset and the model weights are available here: link to figshare repo will be added when it is published. 

The model weights are also available on huggingface: link to hugging face repo will be added when the paper is published. 

Funding: This project was supported by the Viel-Falter Butterfly Monitoring which has received funding from the Federal Ministry for Climate Action, Environment, Energy, Mobility, Innovation and Technology (BMK) and by the project EuroCC Austria which has received funding from the European High Performance Computing Joint Undertaking (JU) and Germany, Bulgaria, Austria, Croatia, Cyprus, Czech Republic, Denmark, Estonia, Finland, Greece, Hungary, Ireland, Italy, Lithuania, Latvia, Poland, Portugal, Romania, Slovenia, Spain, Sweden, France, Netherlands, Belgium, Luxembourg, Slovakia, Norway, Türkiye, Republic of North Macedonia, Iceland, Montenegro, Serbia under grant agreement No 101101903.
We acknowledge the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC supercomputer LUMI, hosted by CSC (Finland) and the LUMI consortium through a EuroHPC Development Access call. 

## Citation 

Citation (bibtex of the associated paper) will be added after publication. 
