======
deoxys
======


.. image:: https://img.shields.io/travis/huynhngoc/deoxys.svg
        :target: https://travis-ci.org/huynhngoc/deoxys

.. image:: https://readthedocs.org/projects/deoxys/badge/?version=latest
        :target: https://deoxys.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://travis-ci.org/huynhngoc/deoxys.svg
   :target: https://travis-ci.org/huynhngoc/deoxys

.. image:: https://coveralls.io/repos/github/huynhngoc/deoxys/badge.svg
   :target: https://coveralls.io/github/huynhngoc/deoxys

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black


Keras-based framework for UNet structure in Cancer Tumor Delineation


* Free software: MIT license
* Documentation: https://deoxys.readthedocs.io.


Features
--------
Applying different deep learning models on medical images


Build from source
--------

Editable mode installation
```
pip install -e .
```

Run test
```
tox .
```

Environment setup
------------------
To run on CPU only (windows)
`set CUDA_VISIBLE_DEVICES=-1`

To use keras in tensorflow (`tf.keras`)
`set KERAS_MODE=TENSORFLOW`

To use keras with multiple backend
`set KERAS_MODE=ALONE`
