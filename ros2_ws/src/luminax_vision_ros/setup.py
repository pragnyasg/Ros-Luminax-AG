import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'luminax_vision_ros'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='LUMI Healthcare Robotics Team',
    maintainer_email='dev@lumi-robotics.org',
    description='ROS 2 Jazzy integration adapter node for LUMI Vision standalone module',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lumi_vision_node = luminax_vision_ros.lumi_vision_node:main',
        ],
    },
)
