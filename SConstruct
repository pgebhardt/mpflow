from testedLibrary import TestedLibrary

# create environment
env = Environment()

# use cuda tools
env.Tool('cuda')

# use clang++
env.Replace(CXX='clang++')

# set flags
env.Append(
    NVCCFLAGS=[
        '-Xcompiler',
        '-fpic',
        '-m64',
        '-arch=sm_30',
        '--compiler-options',
        '-fno-strict-aliasing',
        '-use_fast_math',
        '--ptxas-options=-v',
        '-lineinfo',
        ],
    NVCCINC=[
        '-Iinclude',
        ],
    LIBS=[
        'cudart',
        'cublas',
        'pthread',
        'dl'
        ],
    )

# create librarty
TestedLibrary(name='fasteit',
    env=env, arguments=ARGUMENTS, source_suffix=['cu'])
