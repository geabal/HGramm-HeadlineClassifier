from setuptools import setup,find_packages

setup(
    name='HGramm',           
    version='0.1',            
    install_requires=[],      
    author='Sunjin Park',
    author_email='psjoyo86@gmail.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    package_data={'': ['*.pickle']}
)