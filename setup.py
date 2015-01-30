#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    try:
        from ez_setup import use_setuptools
        use_setuptools()
        from setuptools import setup
    except Exception, e:
        print "Forget setuptools, trying distutils..."
        from distutils.core import setup


description = ("Makes the functions developed in my NEF notebooks importable")
setup(
    name="nef_notebooks",
    version="1.0.0",
    author="Sam Fok",
    author_email="sam.b.fok@gmail.com",
    packages=['neuron',],
    scripts=[],
    url="https://github.com/fragapanagos/notebooks",
    description=description,
    long_description=open('README.md').read(),
    requires=[
        "nengo",
    ],
)
