Experiments around ML experiment configuration

Should allow for easy modification of the config values

Should allow for easy iterations on the config structure (eg add config parameters)

Should even encourage good practices (by making reproducing results easy, encouraging making retrocompatible changes, using typing in code)

Should allow for easy diff between configs (not a difficult problem once dumped to dict)

Current issues:
See binary crossentropy loss, what should be the pattern for choosing between different func/class  
How should list be handeled (eg transforms list), probably not an issue in the code, maybe more of an issue for serializing and diffs  
What pattern for the model (wrapper that has the actual instanciated model as an attribute?, seperate model instanciation and config?)  
  what pattern for the model config: flat config or one config per module with its own params
Should make switching and extending the different components easy (eg data now also returns weights, adding logging to the trainer, adding another evaluation at the end)  

Parametrization of existing class interface may be improved, especially calling get to instanciate the class, better name, better way to go about it (the issue is that some class must be instanciated just before being used because they require eg data or model parameters)

How to dynamically overwrite a parameters after/while creating the config, eg we want to overwrite a a parameter coming for the command line?
- 2 pass, one to create the config, one to instanciate the python objects (modification can then be done inbetween)
- when creating the config object passing additional parameters (eg 'trainer.epochs': 2) and switch dynamically (how exactly would that be implemented? base class with custom init?)
