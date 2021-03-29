from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='swin-transformer-pytorch',
    packages=find_packages(),
    version='0.4.1',
    license='MIT',
    description='Swin Transformer - Pytorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Bernhard Walser',
    author_email='berniwal1997@hotmail.com',
    url='https://github.com/berniwal/swin-transformer-pytorch',
    keywords=[
        'artificial intelligence',
        'attention mechanism',
        'image recognition'
    ],
    install_requires=[
        'torch>=1.8.1',
        'einops>=0.3.0'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6 ',
    ],
)
