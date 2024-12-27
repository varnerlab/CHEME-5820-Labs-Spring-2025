# Lab 1b: Let's get started and configure our environment

## Introduction
This course is for engineers and scientists interested in machine learning and artificial intelligence. 
Engineering practice increasingly relies on computational tools, data analysis, and machine learning approaches. This one-semester course introduces the theoretical foundations of machine learning, focusing on unsupervised, supervised, and online learning and their application in chemical engineering practice. Topics include regularized linear models, boosting, kernel methods, deep learning approaches, generative modeling tools, deterministic and stochastic decision-making, and reinforcement learning approaches. Each week, the course has two lectures, which are theory-based, and two hands-on discussion sessions, which demonstrate the application of the theory to contemporary problems in chemical engineering.

For more information on the course content, policies, procedures, and schedule, see the [2024-2025 Courses of Study](https://classes.cornell.edu/browse/roster/SP25/class/CHEME/5820).

## Requirements
To complete the assignments and projects in this course, you need to install all of these items on your machine.

This course uses [the Julia programming language](https://julialang.org/downloads/) to introduce fundamental concepts of technical computing, data science, machine learning, and artificial intelligence. 
In addition, we use other tools and languages, such as [Python via the Anaconda distribution](https://www.anaconda.com) and [Jupyter Notebooks](https://jupyter.org), [GitHub Desktop](https://desktop.github.com/) for repository management, [GitHub classroom](https://classroom.github.com) for assignments. Finally, we use [Visual Studio Code (VSCode)](https://code.visualstudio.com/download) for development. 

### GitHub Desktop and Anaconda
* GitHub Desktop is a free, open-source application that allows you to work with code hosted on GitHub, such as this course repository. With GitHub Desktop, you can perform Git commands in a graphical user interface, such as committing and pushing changes, rather than using the command line. To download and install GitHub Desktop, follow the [instructions here](https://desktop.github.com/)
* Anaconda is a distribution of the Python and R programming languages for scientific computing that aims to simplify package management and deployment. To install Anaconda on your machine (if you don’t already have a working Python/Jupyter installation), follow the [instructions here](https://www.anaconda.com/download). 

### Julia
`Julia` is a high-level, general-purpose dynamic programming language for numerical analysis and computational science. To download and install the latest version of [Julia](https://julialang.org/downloads/), follow the [instructions here](https://julialang.org/downloads/) for your respective platform.
* For Windows users: During installation, select the `add to path` option to add `Julia` to your search path. You'll need this so that we can start [Julia](https://julialang.org/downloads/) from the terminal in [VSCode](https://code.visualstudio.com/download).
* For macOS users: follow the instructions on the [Julia website for updating the macOS path from the terminal](https://julialang.org/downloads/platform/#optional_add_julia_to_path).

Alternatively, macOS users can manually update your `.zshrc` file in your home directory to include the path entry (edit using Nano or some other text editor): 
```zsh
export PATH=“$PATH:/Applications/Julia-1.10.app/Contents/Resources/Julia/bin”
```

### Visual Studio Code (VSCode)
Visual Studio Code is a free, streamlined code editor that supports development operations such as debugging, task running, and version control. It runs on all major platforms. 
To download and install VSCode, follow the instructions [here](https://code.visualstudio.com/download). Once VSCode is installed, add the Julia language extension via the extensions menu (open the extensions, search for Julia, and install the extension). 

### IJulia
The `IJulia` package must be installed globally so that Jupyter can use `Julia` as a kernel, i.e., we can write `Julia` code in Jupyter notebooks. To install the `IJulia` package, follow these steps:
1. Open the VSCode editor and start a new terminal by selecting the `New Terminal` option from the `Terminal` menu.
1. In the terminal window (zsh for macOS, and PowerShell for Windows) in VSCode, type the command: `julia`
2. Once `Julia` starts, enter package mode by typing the `]` key, and enter the command: `add IJulia`
