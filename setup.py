from setuptools import setup,find_packages


setup(

name="SPARKit",

version="0.1",

packages=find_packages(),

entry_points={

"console_scripts":[

"SPARKit=sparkit.cli:main"

]

}

)