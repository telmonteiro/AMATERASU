from setuptools import setup, find_packages

setup(name = 'AMATERASU',
    version = "0.1.0",
    description = 'AutoMATic Equivalent-width Retrieval for Activity Signal Unveiling',
    url = 'https://github.com/telmonteiro/AMATERASU/',
    license = 'MIT',
    author = 'Telmo Monteiro',
    author_email = 'up202308183@up.pt',
    keywords = ['astronomy', 'activity', 'fits', 'nirps', 'radial velocity', 'exoplanets'],
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.csv'],},
    install_requires = ['numpy', 'pandas', 'astropy', 'matplotlib', 'scipy', 'specutils', 'PyAstronomy', 'tqdm']
)
