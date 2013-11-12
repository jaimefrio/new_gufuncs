from __future__ import division, absolute_import, print_function

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('new_gufuncs', parent_package, top_path)
    config.add_extension('_new_gu_kernels', ['_new_gu_kernels.c.src'])

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)