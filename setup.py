from distutils.core import setup

ext_modules = []
cmdclass = {}

setup(
    name="sgat",
    version="1.0.0",
    description="",
    url="https://github.com/yulun-rayn/sGAT",
    author="Yulun Wu",
    author_email="yulun_wu@berkeley.edu",
    license="MIT",
    packages=["sgat"],
    cmdclass=cmdclass,
    ext_modules=ext_modules
)
