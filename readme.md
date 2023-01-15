Experiments around ML experiment configuration

Should allow for easy modification of the config values
Should allow for easy iterations on the config structure (eg add config parameters)
Should even encourage good practices (by making reproducing results easy, encouraging making retrocompatible changes, using typing in code)
Should allow for easy diff between configs (not a difficult problem once dumped to dict)

Current issues:
See Adam and binary crossentropy loss, what should be the pattern for parametrizing an existing class, what should be the pattern for choosing between different choices
How should list be handeled (eg transforms list), probably not an issue in the code, maybe more of an issue for serializing and diffs
What pattern for the model (wrapper that has the actual instanciated model as an attribute?, seperate model instanciation and config?)
