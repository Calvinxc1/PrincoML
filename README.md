# PrincoML
A few years back I got frustrated with the current derth of Machine Learning libraries out there, namely TensorFlow, PyTorch and Keras. They work great using the standard methods, but I wanted to make something more configurable, where I could customize the architecture, equations, and whatnot, in a much more straightforward method. As such I made this library, PrincoML (named after a good friend of mine).

This is a fantastic library if you're wanting to experiment with novel gradient-based approaches to supervised learning. This is a horrible library to use if you want an easy-to-use out-of-the-box approach to machie learning.

PrincoML allows for a level of flexability beyond what existing libraries provide in how a gradient-descent based machine learning solution can be architectured. It supports simple linear/logistic/GLM style regression, as well as fantastically complex neural networks, and all sorts of solutions inbetween. I've used it both on personal projects and professionally, and it's worked wonders for me. PrincoML is based on PyTorch as the backend, which is a personal preference, as I've tried both Numpy and TensorFlow impementations, and PyTorch just offers a smoother integration for the needs of the library.

Development is largely based around whatever feature I need or feel like implementing. This likely means that CNN/Image Recognition stuff will take a back seat as I'm personally not fond of those approaches, and RNN/Time Series/NLP stuff will take forefront, as that's my speciality (-:

I'm always welcome to accept ideas/code/etc. from other contributors, but the primary purpose of this library is to serve as my own personal machine learning code, that I just happen to share with everyone else (-: As such, enjoy, use at your leisure, and have fun with it!

- Jason Cherry

<JCherry@gmail.com>


# Installation
PrincoML is available through pip: `pip install princoml`.

# Release Notes
Release notes/history is available here: [`release_notes.md`](./release_notes.md).
