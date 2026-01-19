"""
Setup comments:

Install using: pip install -e .
Install the correct version of jax: pip install "jax[cuda12]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Install correct version of tfp: pip install gradient-csmc[tfp]
"""
from pathlib import Path
from setuptools import setup, find_packages


def read_requirements(path: str = "requirements.txt") -> list[str]:
    """
    Read a pip-style requirements file and return a list of requirement strings.

    - Ignores blank lines and comments.
    - Ignores editable (-e) and local path requirements (common in dev workflows).
      If you want to allow those, remove the relevant checks below.
    """
    req_path = Path(__file__).resolve().parent / path
    if not req_path.exists():
        raise FileNotFoundError(f"Could not find {req_path}")

    requirements: list[str] = []
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Optional: skip pip options that setuptools doesn't handle well in install_requires
        if line.startswith(("-r ", "--requirement ", "--find-links ", "-f ", "--index-url ", "--extra-index-url ")):
            continue

        # Optional: skip editable installs / local paths in install_requires
        if line.startswith(("-e ", "--editable ")):
            continue
        if line.startswith((".", "/", "file:")):
            continue

        requirements.append(line)

    return requirements


HERE = Path(__file__).resolve().parent
README = (HERE / "README.md").read_text(encoding="utf-8")

INSTALL_REQUIRES = read_requirements("requirements.txt")

EXTRAS = {
    # Keep optional extras here; these are not typically in requirements.txt
    "tfp": [
        "tensorflow_probability[jax]==0.22.0",
    ],
}

setup(
    name="gradient-csmc",
    version="0.1.1",
    description="Gradient-informed conditional sequential Monte Carlo (JAX).",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Adrien Corenflos",
    url="http://github.com/AdrienCorenflos/mala_csmc",
    license="MIT",
    packages=find_packages(),
    python_requires="==3.12.3",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS,
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "jax",
        "state-space-models",
        "monte-carlo",
        "particle-filter",
        "particle-smoother",
        "particle-mcmc",
        "bayesian",
    ],
)



# """
# Setup comments: 

# Install using: pip install -e .
# Install the correct version of jax: pip install "jax[cuda12]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# Install correct version of tfp: pip install gradient-csmc[tfp] 
# """
# from setuptools import setup, find_packages

# REQUIRED_PACKAGES = [
#     "chex>=0.1.0,<=0.1.85",
#     "matplotlib",
#     "seaborn",
#     "statsmodels",
#     "tqdm",
#     # DO NOT install jax or TFP here — see below.
# ]

# EXTRAS = {
#     # User installs the correct JAX wheel themselves:
#     #   pip install jax==0.4.33
#     #   pip install "jax[cuda12]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#     # "jax": [
#     #     "jax>=0.4.25,<=0.4.34",
#     #     "jaxlib>=0.4.25,<=0.4.34",
#     # ],

#     # TensorFlow Probability only works with specific JAX versions:
#     "tfp": [
#         "tensorflow_probability[jax]==0.22.0",
#     ],
# }

# with open("README.md", "r") as fh:
#     long_description = fh.read()

# setup(
#     name="gradient-csmc",
#     version="0.1.1",
#     description="Gradient-informed conditional sequential Monte Carlo (JAX).",
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     author="Adrien Corenflos",
#     url="http://github.com/AdrienCorenflos/mala_csmc",
#     license="MIT",
#     packages=find_packages(),
#     python_requires=">=3.9",
#     install_requires=REQUIRED_PACKAGES,
#     extras_require=EXTRAS,
#     include_package_data=True,
#     zip_safe=False,
#     keywords=[
#         "jax",
#         "state-space-models",
#         "monte-carlo",
#         "particle-filter",
#         "particle-smoother",
#         "particle-mcmc",
#         "bayesian"
#     ],
# )
