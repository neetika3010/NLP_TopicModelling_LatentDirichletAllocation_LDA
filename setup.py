from setuptools import setup, find_packages

setup(
    name="lda_topic_modelling",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A modular Latent Dirichlet Allocation (LDA) topic modelling pipeline using Gensim, NLTK, and pyLDAvis.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lda-topic-modelling-project",
    packages=find_packages(exclude=["notebooks*", "tests*"]),
    install_requires=[
        "pandas",
        "nltk",
        "gensim",
        "pyLDAvis",
        "matplotlib"
    ],
    entry_points={
        "console_scripts": [
            "lda-run=main:main",  # Allows running pipeline via `lda-run` in terminal
        ],
    },
    python_requires=">=3.8",
)
