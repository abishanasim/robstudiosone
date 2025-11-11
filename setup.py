# project_seeder/setup.py
import os
from glob import glob
from setuptools import setup

package_name = 'project_seeder'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install any launch files that actually exist in your source tree
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # (Optional) install configs if you have them
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='',
    description='Waypoints for the seeder project',
    license='',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'tree_goals = project_seeder.tree_goals:main',
        'colour_detection = project_seeder.colour_detection:main',
        'husky_seeder = project_seeder.husky_seeder:main'
    ],
},

)
