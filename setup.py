from setuptools import setup

__version__ = (0, 0, 0)

setup(
    name="pu-ray",
    description="PU-Ray: Point Cloud Upsampling via Ray Marching on Implicit Surface",
    version=".".join(str(d) for d in __version__),
    author="Sangwon Lim",
    author_email="sangwon3@ualberta.ca",
    packages=["pu_ray"],
    include_package_data=True,
    scripts="""
        ./scripts/pu_ray
        ./scripts/train_model
        ./scripts/test_model
        ./scripts/evaluate
        ./scripts/generate_gt
    """.split(),
)
