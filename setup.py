from setuptools import setup

setup(
    name="psoc",
    version="0.1.0",
    description="Stochastic Optimal Control via Conditional Sequential Monte Carlo",
    author="Hany Abdulsamad",
    author_email="hany@robot-learning.de",
    install_requires=[
        "numpy",
        "scipy",
        "jax",
        "jaxlib",
        "jaxopt",
        "typing_extensions",
        "matplotlib",
        "optax",
        "flax",
        "distrax"
    ],
    packages=["psoc"],
    zip_safe=False,
)
