import os
from setuptools import find_packages, setup

# 1. Dynamically find the version from the source
def get_version(filename):
    import ast
    with open(filename) as f:
        for line in f:
            if line.startswith("__version__"):
                return ast.parse(line).body[0].value.s
    return "6.2.0" # Fallback

version = get_version("src/gym_duckietown/__init__.py")

# 2. Read requirements.txt for a single "Source of Truth"
# This ensures pip install -e . installs everything in your requirements file
with open("requirements.txt") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="duckie-rl",
    version=version,
    description="Modified Duckietown for SAC/TD3 Reinforcement Learning Using CleanRL",
    
    # --- THIS IS THE KEY CHANGE ---
    # We tell Python: 
    # 1. The main code is in 'src'
    # 2. But 'rl', 'utils', and 'cleanrl_utils' are in the root ('.')
    package_dir={
        "": "src",
        "rl": "rl",
        "utils": "utils",
        "path_planning": "path_planning",
        "cleanrl_utils": "cleanrl_utils"
    },
    
    # Now find_packages will look in 'src', 
    # and the others will be pulled from the root mapping above
    packages=find_packages("src") + ["rl", "utils", "path_planning", "cleanrl_utils"],
    
    zip_safe=False,
    include_package_data=True,
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "dt-check-gpu=gym_duckietown.check_hw:main",
        ],
    },
)