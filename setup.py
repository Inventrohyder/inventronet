from setuptools import setup, find_packages

package_name = "inventronet"
package_version = "0.0.4-alpha"
package_description = "A package for building and testing neural networks"

# Read the README.md file
with open("README.md", "r", encoding="utf-8") as f:
    package_long_description = f.read()

package_url = "https://github.com/inventrohyder/inventronet"
package_author = "inventrohyder"
package_author_email = "haitham.hyder@uni.minerva.edu"
package_license = "MIT"
package_classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
package_keywords = "neural network, machine learning, artificial intelligence"
package_install_requires = [
    "numpy",
]

package_extras_require = {
    "test": ["pytest", "pytest-cov"],
    "docs": ["sphinx", "sphinx-rtd-theme"],
    "lint": ["flake8", "ruff", "black"],
}

setup(
    name=package_name,
    version=package_version,
    description=package_description,
    long_description=package_long_description,
    long_description_content_type="text/markdown",
    url=package_url,
    author=package_author,
    author_email=package_author_email,
    license=package_license,
    packages=find_packages(),
    install_requires=package_install_requires,
    classifiers=package_classifiers,
    extras_require=package_extras_require,
    python_requires=">=3.9",
)
